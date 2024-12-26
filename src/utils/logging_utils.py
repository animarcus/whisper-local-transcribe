import logging
import sys
from typing import Optional, Union
import colorama
from colorama import Fore, Back, Style
from enum import Enum

# Initialize colorama
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

class LogLevel(Enum):
    HIGHLIGHT = "highlight"
    DETAIL = "detail"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PRIMARY = "primary"
    DEBUG = "debug"

class LogMessage:
    def __init__(self, message: str, level: LogLevel = LogLevel.DETAIL):
        self.message = message
        self.level = level

    @staticmethod
    def highlight(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.HIGHLIGHT)
    
    @staticmethod
    def detail(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.DETAIL)
    
    @staticmethod
    def success(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.SUCCESS)
    
    @staticmethod
    def warning(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.WARNING)
    
    @staticmethod
    def error(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.ERROR)
    
    @staticmethod
    def primary(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.PRIMARY)
    
    @staticmethod
    def debug(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.DEBUG)

class ColorfulTranscriptionLogger:
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger('transcription')
        self.logger.setLevel(level)
        
        # Console handler with colored output
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        self.logger.addHandler(console)
        
        # Add file handler if specified (without color codes)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            # Strip color codes for file output
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def highlight(self, msg: str):
        """For important progress updates - Bright cyan with black background"""
        self.logger.info(f"{COLORS['HIGHLIGHT']}{msg}")
    
    def detail(self, msg: str):
        """For detailed information - Bright white"""
        self.logger.info(f"{COLORS['TEXT']}{msg}")
    
    def success(self, msg: str):
        """For successful operations - Bright green"""
        self.logger.info(f"{COLORS['SUCCESS']}{msg}")
    
    def warning(self, msg: str):
        """For warnings - Bright yellow"""
        self.logger.warning(f"{COLORS['WARNING']}{msg}")
    
    def error(self, msg: str):
        """For errors - Bright red"""
        self.logger.error(f"{COLORS['ERROR']}{msg}")
    
    def primary(self, msg: str):
        """For primary information - Bright cyan"""
        self.logger.info(f"{COLORS['PRIMARY']}{msg}")
    
    def debug(self, msg: str):
        """For debug information - Normal intensity white"""
        self.logger.debug(msg)
    
    def log(self, message: Union[str, LogMessage]) -> None:
        """Log a message with appropriate level"""
        if isinstance(message, str):
            self.detail(message)
        else:
            getattr(self, message.level.value)(message.message)

_logger_instance = None

def setup_logging(path: str = None, level: int = logging.INFO) -> ColorfulTranscriptionLogger:
    """Initialize or return the logger instance"""
    global _logger_instance
    if _logger_instance is None:
        log_file = f"{path}/transcription.log" if path else None
        _logger_instance = ColorfulTranscriptionLogger(log_file, level)
    return _logger_instance

def get_logger() -> ColorfulTranscriptionLogger:
    """Get the current logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ColorfulTranscriptionLogger()
    return _logger_instance
