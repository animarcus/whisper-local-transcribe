import argparse
import os

from src._LocalTranscribe import transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio from video files using Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing video files.")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v2", 
                    choices=[
                        'openai/whisper-tiny', 
                        'openai/whisper-base', 
                        'openai/whisper-small', 
                        'openai/whisper-medium', 
                        'openai/whisper-large-v2',
                        'openai/whisper-large-v3',  # Add latest
                        'distil-whisper/distil-large-v2'  # Add faster alternative
                    ],
                    help="Model to use for transcription.")
    parser.add_argument("--language", type=str, default=None, 
                        help="Language of the audio (or leave empty to auto-detect).")
    parser.add_argument("--verbose", action="store_true", 
                        help="Output transcription to terminal.")
    
    args = parser.parse_args()
    
    # Resolve relative path
    folder_path = os.path.abspath(args.folder)
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Start transcription
    try:
        output_text: str = transcribe(folder_path, args.model, args.language, args.verbose)
        print("Transcription completed successfully.")
        if args.verbose:
            print(output_text)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")

if __name__ == "__main__":
    main()
