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
    'PRIMARY': Fore.CYAN + Style.BRIGHT,                    
    'SUCCESS': Fore.GREEN + Style.BRIGHT,                   
    'WARNING': Fore.YELLOW + Style.BRIGHT,                  
    'ERROR': Fore.RED + Style.BRIGHT,                      
    'TEXT': Fore.WHITE + Style.NORMAL,                     
    'HIGHLIGHT': Fore.WHITE + Back.BLUE + Style.BRIGHT,    
    'INFO': Fore.CYAN + Style.DIM,                         
    'SUBTLE': Fore.WHITE + Style.DIM,                      
    'METRIC': Fore.BLACK + Back.GREEN + Style.BRIGHT,      
    'STATS': Fore.BLACK + Back.GREEN + Style.NORMAL,       
    'HEADER': Fore.WHITE + Back.BLUE + Style.BRIGHT,       
    'SUMMARY': Fore.WHITE + Back.BLUE + Style.BRIGHT,      
}

class LogLevel(Enum):
    HIGHLIGHT = "highlight"
    DETAIL = "detail"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PRIMARY = "primary"
    DEBUG = "debug"
    INFO = "info"
    SUBTLE = "subtle"
    METRIC = "metric"
    STATS = "stats"
    HEADER = "header"
    SUMMARY = "summary"

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
    
    @staticmethod
    def info(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.INFO)
    
    @staticmethod
    def subtle(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.SUBTLE)
    
    @staticmethod
    def metric(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.METRIC)
    
    @staticmethod
    def stats(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.STATS)
    
    @staticmethod
    def header(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.HEADER)
    
    @staticmethod
    def summary(msg: str) -> 'LogMessage':
        return LogMessage(msg, LogLevel.SUMMARY)

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
    
    def _log_with_reset(self, color: str, msg: str):
        """Helper method to ensure proper color reset"""
        self.logger.info(f"{color}{msg}{Style.RESET_ALL}")
    
    def highlight(self, msg: str):
        """For important progress updates"""
        self._log_with_reset(COLORS['HIGHLIGHT'], msg)
    
    def detail(self, msg: str):
        """For detailed information"""
        self._log_with_reset(COLORS['TEXT'], msg)
    
    def success(self, msg: str):
        """For successful operations"""
        self._log_with_reset(COLORS['SUCCESS'], msg)
    
    def warning(self, msg: str):
        """For warnings"""
        self._log_with_reset(COLORS['WARNING'], msg)
    
    def error(self, msg: str):
        """For errors"""
        self._log_with_reset(COLORS['ERROR'], msg)
    
    def primary(self, msg: str):
        """For primary information"""
        self._log_with_reset(COLORS['PRIMARY'], msg)
    
    def debug(self, msg: str):
        """For debug information"""
        self._log_with_reset(COLORS['TEXT'], msg)
    
    def info(self, msg: str):
        """For informational messages"""
        self._log_with_reset(COLORS['INFO'], msg)
    
    def subtle(self, msg: str):
        """For discrete updates"""
        self._log_with_reset(COLORS['SUBTLE'], msg)
    
    def metric(self, msg: str):
        """For metrics and numbers"""
        self._log_with_reset(COLORS['METRIC'], msg)
    
    def stats(self, msg: str):
        """For performance statistics"""
        self._log_with_reset(COLORS['STATS'], msg)
    
    def header(self, msg: str):
        """For section headers"""
        self._log_with_reset(COLORS['HEADER'], msg)
    
    def summary(self, msg: str):
        """For summary sections"""
        self._log_with_reset(COLORS['SUMMARY'], msg)

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
