import os
import datetime

def save_transcription(path, title, segments):
    """Save transcription to a file with timestamps"""
    try:
        os.makedirs(os.path.join(path, 'transcriptions'), exist_ok=True)
    except FileExistsError:
        pass
    
    transcription_file = os.path.join(path, "transcriptions", f"{title}-transcription.txt")
    
    with open(transcription_file, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(f'[{segment["start"]} --> {segment["end"]}] {segment["text"]}\n')
        file.write('\n# END OF TRANSCRIPTION')
