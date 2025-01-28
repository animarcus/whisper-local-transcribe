# src/utils/performance_utils.py
from typing import List
import psutil
import torch
import platform
import time
from datetime import datetime
import wave
import contextlib
from moviepy.editor import VideoFileClip
from src.utils.logging_utils import LogMessage, LogLevel

def get_system_info() -> List[LogMessage]:
    """Get basic system performance statistics"""
    try:
        # CPU Info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)  # Convert to GB
        
        # GPU Info if available
        gpu_info = "No GPU detected"
        if torch.backends.mps.is_available():  # For Mac
            gpu_info = "Apple Silicon GPU (MPS) available"
        elif torch.cuda.is_available():  # For NVIDIA
            gpu_info = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
            gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
            gpu_info += f" (Using {gpu_memory:.2f} GB)"
            
        return [
            LogMessage.header("System Performance:"),
            LogMessage.stats(f"CPU Usage: {cpu_percent}%"),
            LogMessage.stats(f"RAM Usage: {memory_used_gb:.2f} GB / {memory.total / (1024 ** 3):.2f} GB"),
            LogMessage.stats(f"GPU: {gpu_info}")
        ]
        
    except Exception as e:
        return [LogMessage.error(f"Error getting system info: {str(e)}")]

def optimize_chunk_size():
    """
    Determine optimal chunk size based on system resources.
    Returns recommended chunk size in seconds.
    """
    try:
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 ** 3)
        
        # Basic heuristic for chunk size:
        # - For systems with less than 8GB RAM: 15 seconds
        # - For systems with 8-16GB RAM: 30 seconds
        # - For systems with more than 16GB RAM: 45 seconds
        # - If GPU is available, we can use larger chunks
        
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            if total_memory_gb < 8: return 20
            elif total_memory_gb < 16: return 40
            else: return 60
        else:
            if total_memory_gb < 8: return 15
            elif total_memory_gb < 16: return 30
            else: return 45
                
    except Exception:
        return 30  # Default safe value

def get_audio_duration(file_path):
    """Get duration of audio/video file in seconds"""
    try:
        if file_path.lower().endswith('.mp4'):
            with VideoFileClip(file_path) as video:
                return video.duration
        elif file_path.lower().endswith('.wav'):
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        else:
            return None
    except Exception as e:
        return None

def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time with appropriate units"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {int(remaining_seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {int(remaining_seconds)}s"

class TranscriptionTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def get_elapsed(self) -> float:
        """Get elapsed time without stopping the timer"""
        if not self.start_time:
            return 0.0
        if self.end_time:
            return self.duration
        return time.time() - self.start_time
    
    def get_performance_stats(self, file_path: str, chunk_size: int) -> List[LogMessage]:
        """Calculate and format performance statistics"""
        if not self.duration:
            return [LogMessage.warning("Timer not stopped")]
            
        audio_duration = get_audio_duration(file_path)
        if audio_duration is None:
            return [LogMessage.metric(f"Processing Time: {self.duration:.2f} seconds")]

        realtime_ratio = self.duration / audio_duration
        chunks_processed = audio_duration / chunk_size
        avg_chunk_time = self.duration / chunks_processed
        
        return [
            LogMessage.header("📊 Transcription Performance:"),
            LogMessage.metric(f"\tAudio Duration: {audio_duration:.2f} seconds"),
            LogMessage.metric(f"\tProcessing Time: {self.duration:.2f} seconds"),
            LogMessage.metric(f"\tRealtime Ratio: {realtime_ratio:.2f}x realtime"),
            LogMessage.stats(f"\tChunks Processed: {int(chunks_processed)}"),
            LogMessage.stats(f"\tAverage Time per Chunk: {avg_chunk_time:.2f} seconds")
        ]


    def get_debug_info(self, file_path):
        """Get debug information about the transcription"""
        audio_duration = get_audio_duration(file_path)
        return [LogMessage.debug(f"Debug - Audio duration: {audio_duration}, File: {file_path}")]
