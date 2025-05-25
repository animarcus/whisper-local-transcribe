import datetime
import os
import tempfile
import warnings

import torch
from datasets import Audio, Dataset
from moviepy.editor import VideoFileClip
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.utils.file_utils import get_path
from src.utils.logging_utils import setup_logging
from src.utils.performance_utils import (
    TranscriptionTimer,
    format_elapsed_time,
    get_system_info,
    optimize_chunk_size,
)
from src.utils.transcription_utils import save_transcription

warnings.filterwarnings("ignore", message="Due to a bug fix in")
warnings.filterwarnings("ignore", message="Passing a tuple of")

END_OF_TRANSCRIPTION_MARKER = "# END OF TRANSCRIPTION"



def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file and save it temporarily"""
    video = VideoFileClip(video_path)
    audio = video.audio
    
    # Create temporary wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio.write_audiofile(temp_file.name, logger=None)
        video.close()
        return temp_file.name

def format_timestamp(seconds) -> str:
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=round(seconds, 2)))

def format_timestamp_with_minutes(seconds) -> str:
    """Convert seconds to 'XmYs (Z seconds)' format"""
    minutes = int(seconds // 60)
    remaining_seconds = round(seconds % 60)
    return f"{minutes}m{remaining_seconds}s ({round(seconds, 2)} seconds)"

def format_chunk_progress(file_num, total_files, title, timestamp_start, timestamp_end, total_duration=None) -> str:
    """Format the progress string for each chunk"""
    progress = f"[File {file_num}/{total_files}] {title}"
    timestamp = f"[{timestamp_start} --> {timestamp_end}]"
    if total_duration is not None:
        return f"{progress}\n{timestamp} (Total: {format_timestamp(total_duration)})"
    return f"{progress}\n{timestamp}"



def transcribe(path, model_name=None, language="en", verbose=False) -> str:
    """
    Transcribes audio files in a specified folder using OpenAI's Whisper model.
    """
    logger = setup_logging(path)
    
    valid_extensions = ('.mp4', '.wav', '.mp3')  # Add any other extensions you want to support
    glob_file = [f for f in get_path(path) if os.path.isfile(f) and f.lower().endswith(valid_extensions)]
    
    if not glob_file:
        return 'No valid audio or video files found in the specified folder.'
    
    # Filter out files that have already been transcribed
    files_to_transcribe = []
    skipped_files = []
    for file in glob_file:
        title = os.path.basename(file).split('.')[0]
        transcription_file = os.path.join(path, "transcriptions", f"{title}-transcription.txt")
        
        if os.path.exists(transcription_file):
            with open(transcription_file, 'r') as f:
                if END_OF_TRANSCRIPTION_MARKER in f.read():
                    skipped_files.append({
                        'title': title,
                        'reason': 'Already transcribed'
                    })
                    continue
        
        files_to_transcribe.append({
                'path': file,
                'title': os.path.basename(file).split('.')[0],
                'transcription_file': os.path.join(path, "transcriptions", f"{title}-transcription.txt"),
            }
        )

    total_files_to_transcribe = len(files_to_transcribe)

    # Print initial summary
    if verbose:
        logger.header("\n══════════════════════ File Analysis ══════════════════════")
        logger.metric(f"Found {len(glob_file)} files")
        
        if skipped_files:
            logger.info(f"\nSkipping {len(skipped_files)} already transcribed files:")
            for skipped_file in skipped_files:
                logger.warning(f"  ● {skipped_file['title']}")  # Using bullet point for better visual
        
        logger.info(f"\n{total_files_to_transcribe} Files to transcribe:")
        for file in files_to_transcribe:
            logger.primary(f"  ▶ {file['title']}")

    if not files_to_transcribe:
        return 'All files have already been transcribed.'

    if verbose:
        chunk_length = optimize_chunk_size()
        logger.header("\n══════════════════════ System Analysis ══════════════════════")
        logger.stats(f"Using chunk size of {format_timestamp_with_minutes(chunk_length)} based on system configuration")
        system_info = get_system_info()
        for message in system_info:
            logger.log(message)
    else:
        chunk_length = 30  # Default value if not verbose

    # Check for GPU acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            logger.success(f"Using NVIDIA GPU acceleration: {torch.cuda.get_device_name(0)} ✨")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            logger.success("Using Apple Silicon GPU acceleration ✨")
    else:
        device = torch.device("cpu")
        if verbose:
            logger.info("Using CPU for processing")
    
    if verbose:
        logger.header("\n══════════════════════ Model Configuration ══════════════════════")
        logger.subtle(f"Loading {model_name or 'default'} model configuration...")
        logger.info(f"Device: {device.type.upper()}")
        if device.type == "mps":
            logger.success("Using Apple Silicon GPU acceleration ✨")
        elif device.type == "cuda":
            logger.success(f"Using NVIDIA GPU acceleration: {torch.cuda.get_device_name(0)} ✨")
        else:
            logger.info("Using CPU for processing")

    # Load model and processor
    if verbose:
        logger.subtle("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(model_name)
    
    if verbose:
        logger.subtle("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        use_cache=True,
        return_dict=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None
    ).to(device)
    
    if device.type == "cuda":
        # Optimize for 8GB VRAM (Clear memory between files)
        torch.cuda.empty_cache()
        if verbose:
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            logger.info("Applied float16 optimization for VRAM efficiency")
    
    if verbose:
        logger.subtle("Configuring model parameters...")
    # Set up forced decoder IDs for transcription (not translation)
    if language:
        if verbose:
            logger.subtle(f"Setting up forced transcription for language: {language}")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.config.suppress_tokens = None
    else:
        if verbose:
            logger.subtle("Setting up language auto-detection")
        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids

    if verbose:
        logger.subtle("Model initialization complete! ✨")
        logger.subtle("\nStarting transcription process...")


    # Define generate function for consistent transcription parameters
    def generate_transcription(model, chunk_input, chunk_attention_mask):
        return model.generate(
            chunk_input,
            attention_mask=chunk_attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=448,
            use_cache=False  # Explicitly disable caching
        )

    # Start main loop
    files_transcripted = []
    transcription_summary = []
    temp_files = []  # Keep track of temporary files to clean up later

    # Start timer for overall time
    overall_timer = TranscriptionTimer()
    overall_timer.start()
    
    # Start transcriptions
    for i, file in enumerate(files_to_transcribe, 1):
        file_path = file['path']
        title = file['title']
        transcription_file = file['transcription_file']
        
        file_timer = TranscriptionTimer()  # Create a timer for this specific file
        file_timer.start()
        
        if verbose:
            logger.header(f"\nStarting transcription of: {title} [{i}/{total_files_to_transcribe}] 🕐")
        
        try:
            # Handle MP4 files
            if file_path.lower().endswith('.mp4'):
                if verbose:
                    logger.subtle("Extracting audio from video file...")
                audio_file = extract_audio_from_video(file_path)
                temp_files.append(audio_file)
            else:
                audio_file = file_path
                
            # Create dataset with single audio file
            if verbose:
                logger.subtle("Creating audio dataset...")
            dataset = Dataset.from_dict({"audio": [audio_file]}).cast_column("audio", Audio(sampling_rate=16000))
            
            # Load audio and preprocess
            if verbose:
                logger.subtle("Processing audio features...")
            audio_data = dataset[0]["audio"]
            input_features = processor(
                audio_data["array"],
                sampling_rate=audio_data["sampling_rate"],
                return_tensors="pt"
            ).input_features.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_features)
            
            # Generate transcription with chunking
            stride_length = chunk_length // 6
            total_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            original_duration = total_duration
            segments = []
            
            if verbose:
                logger.header("\n══════════════════════ Processing Configuration ══════════════════════")
                logger.metric(f"Chunk size:          {format_timestamp_with_minutes(chunk_length)}")
                logger.metric(f"Stride length:       {format_timestamp_with_minutes(stride_length)}")
                logger.metric(f"Total audio duration: {format_timestamp_with_minutes(total_duration)}")
                logger.metric(f"Expected chunks:      {int(total_duration / (chunk_length - stride_length))}")

            
            logger.detail(f"File duration: {format_timestamp(total_duration)}")
            
            for chunk_start in range(0, int(total_duration), chunk_length - stride_length):
                chunk_end = min(chunk_start + chunk_length, total_duration)
                
                chunk_start_sample = int(chunk_start * audio_data["sampling_rate"])
                chunk_end_sample = int(chunk_end * audio_data["sampling_rate"])
                chunk_input = processor(
                    audio_data["array"][chunk_start_sample:chunk_end_sample],
                    sampling_rate=audio_data["sampling_rate"],
                    return_tensors="pt"
                ).input_features.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
                
                chunk_attention_mask = torch.ones_like(chunk_input)
                
                with torch.no_grad():
                    outputs = generate_transcription(
                        model,
                        chunk_input,
                        chunk_attention_mask
                    )
                
                # Decode and get timestamps
                chunk_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
                
                if chunk_text.strip():  # Only add non-empty segments
                    segment = {
                        "start": format_timestamp(chunk_start),
                        "end": format_timestamp(chunk_end),
                        "text": chunk_text.strip()
                    }
                    segments.append(segment)
                    
                    if verbose:
                        progress = format_chunk_progress(
                            i, 
                            total_files_to_transcribe, 
                            title, 
                            segment["start"], 
                            segment["end"], 
                            total_duration
                        )
                        overall_elapsed = format_elapsed_time(overall_timer.get_elapsed())
                        file_elapsed = format_elapsed_time(file_timer.get_elapsed())
                        logger.primary(f"\n{progress} \t| Elapsed Time - Total: {overall_elapsed} | File: {file_elapsed}")
                        logger.detail(f"{segment['text']}\n")
            
            # After processing is complete
            file_timer.stop()
            transcription_time = file_timer.duration
            transcription_ratio = transcription_time / original_duration
            
            if verbose:
                # Handle performance stats
                logger.header("---\nTranscription done!\n---")
                stats_lines = file_timer.get_performance_stats(file_path, chunk_length)
                for log_message in stats_lines:
                    logger.log(log_message)
            
            # Save the transcription
            save_transcription(path, title, segments)
            
            # Add to summary and transcribed files list
            transcription_summary.append({
                'title': title,
                'status': 'completed',
                'duration': transcription_time,
                'original_duration': original_duration,
                'transcription_ratio': transcription_ratio,
                'file_number': i,
                'total_files': total_files_to_transcribe
            })
            files_transcripted.append(title)
            
        except RuntimeError as e:
            file_timer.stop()
            error_msg = 'Not a valid file, skipping.'
            logger.error(f'Error processing file: {str(e)}')
            logger.error(error_msg)
            transcription_summary.append({
                'title': title,
                'status': 'failed',
                'error': error_msg,
                'duration': file_timer.duration,
                'file_number': i,
                'total_files': total_files_to_transcribe
            })
            continue
        except Exception as e:
            file_timer.stop()
            error_msg = f'Unexpected error: {str(e)}'
            logger.error(error_msg)
            transcription_summary.append({
                'title': title,
                'status': 'failed',
                'error': error_msg,
                'duration': file_timer.duration,
                'file_number': i,
                'total_files': total_files_to_transcribe
            })
            continue
        finally:
            # Clean up temporary files
            if file_path.lower().endswith('.mp4') and 'audio_file' in locals():
                try:
                    os.unlink(audio_file)
                except:
                    pass

    # Print final summary
    logger.header("="*50)
    logger.summary("TRANSCRIPTION SUMMARY")
    logger.metric(f"Total Elapsed Time: {format_elapsed_time(overall_timer.get_elapsed())}")
    logger.header("="*50)

    # Show successful transcriptions with timing
    successful_transcriptions = [f for f in transcription_summary if f['status'] == 'completed']
    if successful_transcriptions:
        logger.header("\nSuccessfully Transcribed Files:")
        for file_info in successful_transcriptions:
            logger.success(f"- {file_info['title']}: {format_timestamp_with_minutes(file_info['duration'])} (Original: {format_timestamp_with_minutes(file_info['original_duration'])}, Ratio: {file_info['transcription_ratio']:.2f})")

    # Show skipped files
    if skipped_files:
        logger.header("Skipped Files (Already Transcribed):")
        for file_info in skipped_files:
            logger.warning(f"- {file_info['title']}")  # Modified to show just the title

    # Show failed files
    failed_transcriptions = [f for f in transcription_summary if f['status'] == 'failed']
    if failed_transcriptions:
        logger.error("Failed Transcriptions:")
        for file_info in failed_transcriptions:
            logger.error(f"- {file_info['title']}: {file_info['error']}")

    # Show overall statistics
    total_time = sum(f['duration'] for f in transcription_summary)
    logger.header("="*50)
    logger.summary("Overall Statistics:")
    logger.metric(f"Total processing time: {format_timestamp_with_minutes(total_time)}")
    logger.stats(f"Successfully transcribed: {len(successful_transcriptions)} files")
    logger.stats(f"Skipped: {len(skipped_files)} files")
    logger.stats(f"Failed: {len(failed_transcriptions)} files")
    logger.header("="*50)
    
    return 'Transcription process completed.'
