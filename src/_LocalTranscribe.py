import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, Dataset
import colorama
from colorama import Back, Fore, Style
from src.utils.file_utils import get_path
from src.utils.transcription_utils import save_transcription
from src.utils.performance_utils import TranscriptionTimer
from moviepy.editor import VideoFileClip
import tempfile
import soundfile as sf
import datetime

colorama.init(autoreset=True)

# Color configuration
COLORS = {
    'PRIMARY': Fore.CYAN + Style.BRIGHT,     # Bright cyan for better visibility
    'SUCCESS': Fore.GREEN + Style.BRIGHT,    # Bright green
    'WARNING': Fore.YELLOW + Style.BRIGHT,   # Bright yellow
    'ERROR': Fore.RED + Style.BRIGHT,        # Bright red
    'TEXT': Style.BRIGHT,                    # Bright white (default color)
    'HIGHLIGHT': Back.BLACK + Fore.CYAN + Style.BRIGHT,  # Black background with bright cyan text
}
END_OF_TRANSCRIPTION_MARKER = "# END OF TRANSCRIPTION"

def extract_audio_from_video(video_path):
    """Extract audio from video file and save it temporarily"""
    video = VideoFileClip(video_path)
    audio = video.audio
    
    # Create temporary wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio.write_audiofile(temp_file.name, logger=None)
        video.close()
        return temp_file.name

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=round(seconds, 2)))

def format_timestamp_with_minutes(seconds):
    """Convert seconds to 'XmYs (Z seconds)' format"""
    minutes = int(seconds // 60)
    remaining_seconds = round(seconds % 60)
    return f"{minutes}m{remaining_seconds}s ({round(seconds, 2)} seconds)"

def format_chunk_progress(file_num, total_files, title, timestamp_start, timestamp_end, total_duration=None):
    """Format the progress string for each chunk"""
    progress = f"[{file_num}/{total_files} {title}] [{timestamp_start} --> {timestamp_end}]"
    if total_duration is not None:
        progress += f" (Total duration: {format_timestamp(total_duration)})"
    return progress

def transcribe(path, model_name=None, language="en", verbose=False):
    """
    Transcribes audio files in a specified folder using OpenAI's Whisper model.
    """
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
        print(f"\n{COLORS['PRIMARY']}Found {len(glob_file)} files:")
        
        if skipped_files:
            print(f"{COLORS['WARNING']}Skipping {len(skipped_files)} already transcribed files:")
            for skipped_file in skipped_files:
                print(f"{COLORS['WARNING']}\t- {skipped_file['title']}")
        print(f"{COLORS['TEXT']}{total_files_to_transcribe} Files to transcribe:")
        for file in files_to_transcribe:
            print(f"{COLORS['PRIMARY']}\t- {file['title']}")
        print("\n")

    if not files_to_transcribe:
        return 'All files have already been transcribed.'

    if verbose:
        from src.utils.performance_utils import get_system_info, optimize_chunk_size
        print(get_system_info())
        chunk_length = optimize_chunk_size()
        print(f"{COLORS['PRIMARY']}Using chunk size of {chunk_length} seconds based on your system\n")
    else:
        chunk_length = 30  # Default value if not verbose

    # Check for GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        use_cache=True,
        return_dict=True,
    ).to(device)
    
    # Set up forced decoder IDs for transcription (not translation)
    if language:
        # If language is specified, use it
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.config.suppress_tokens = None  # Add this line
    else:
        # If no language specified, let the model detect it (don't force English)
        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids


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

    
    # Start transcriptions
    for i, file in enumerate(files_to_transcribe, 1):
        file_path = file['path']
        title = file['title']
        transcription_file = file['transcription_file']
        
        file_timer = TranscriptionTimer()  # Create a timer for this specific file
        file_timer.start()
        
        if verbose:
            print(COLORS['HIGHLIGHT'] + f'\nStarting transcription of: {title} [{i}/{total_files_to_transcribe}] 🕐\n')
        
        try:
            # Handle MP4 files
            if file_path.lower().endswith('.mp4'):
                audio_file = extract_audio_from_video(file_path)
                temp_files.append(audio_file)
            else:
                audio_file = file_path
                
            # Create dataset with single audio file
            dataset = Dataset.from_dict({"audio": [audio_file]}).cast_column("audio", Audio(sampling_rate=16000))
            
            # Load audio and preprocess
            audio_data = dataset[0]["audio"]
            input_features = processor(
                audio_data["array"],
                sampling_rate=audio_data["sampling_rate"],
                return_tensors="pt"
            ).input_features.to(device)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_features)
            
            # Generate transcription with chunking
            stride_length = chunk_length // 6
            total_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            original_duration = total_duration
            segments = []
            
            print(f"\n{COLORS['TEXT']}File duration: {format_timestamp(total_duration)}")
            
            for chunk_start in range(0, int(total_duration), chunk_length - stride_length):
                chunk_end = min(chunk_start + chunk_length, total_duration)
                
                chunk_start_sample = int(chunk_start * audio_data["sampling_rate"])
                chunk_end_sample = int(chunk_end * audio_data["sampling_rate"])
                chunk_input = processor(
                    audio_data["array"][chunk_start_sample:chunk_end_sample],
                    sampling_rate=audio_data["sampling_rate"],
                    return_tensors="pt"
                ).input_features.to(device)
                
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
                        print(COLORS['PRIMARY'] + progress)  # Changed from GREEN to BLUE for better visibility
                        print(COLORS['TEXT'] + f'{segment["text"]}\n')
            
            # After processing is complete
            file_timer.stop()
            transcription_time = file_timer.duration
            transcription_ratio = transcription_time / original_duration
            
            if verbose:
                print(f"\n{COLORS['PRIMARY']}Processing completed for file {i}/{total_files_to_transcribe}")
                stats = file_timer.get_performance_stats(file_path, chunk_length)
                print(stats.replace(f"{chunk_length} seconds", format_timestamp_with_minutes(chunk_length)))
            
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
            print(COLORS['ERROR'] + f'Error processing file: {str(e)}')
            print(COLORS['ERROR'] + error_msg)
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
            print(COLORS['ERROR'] + error_msg)
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
    print("\n" + "="*50)
    print(f"{COLORS['PRIMARY']}TRANSCRIPTION SUMMARY")
    print("="*50)

    # Show successful transcriptions with timing
    successful_transcriptions = [f for f in transcription_summary if f['status'] == 'completed']
    if successful_transcriptions:
        print(f"\n{COLORS['SUCCESS']}Successfully Transcribed Files:")
        for file_info in successful_transcriptions:
            print(f"{COLORS['SUCCESS']}- {file_info['title']}: {format_timestamp_with_minutes(file_info['duration'])} (Original: {format_timestamp_with_minutes(file_info['original_duration'])}, Ratio: {file_info['transcription_ratio']:.2f})")

    # Show skipped files
    if skipped_files:
        print(f"\n{COLORS['WARNING']}Skipped Files (Already Transcribed):")
        for file_info in skipped_files:
            print(f"{COLORS['WARNING']}- {file_info['title']}")  # Modified to show just the title

    # Show failed files
    failed_transcriptions = [f for f in transcription_summary if f['status'] == 'failed']
    if failed_transcriptions:
        print(f"\n{COLORS['ERROR']}Failed Transcriptions:")
        for file_info in failed_transcriptions:
            print(f"{COLORS['ERROR']}- {file_info['title']}: {file_info['error']}")

    # Show overall statistics
    total_time = sum(f['duration'] for f in transcription_summary)
    print("\n" + "="*50)
    print(f"{COLORS['PRIMARY']}Overall Statistics:")
    print(f"{COLORS['TEXT']}Total processing time: {format_timestamp_with_minutes(total_time)}")
    print(f"Successfully transcribed: {len(successful_transcriptions)} files")
    print(f"Skipped: {len(skipped_files)} files")
    print(f"Failed: {len(failed_transcriptions)} files")
    print("="*50)
    
    return 'Transcription process completed.'
