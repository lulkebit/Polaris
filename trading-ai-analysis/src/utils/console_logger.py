import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path
import colorama
from colorama import Fore, Style

class ConsoleLogger:
    """
    Ein Logger für die Konsolen-Ausgabe mit farbiger Formatierung und
    übersichtlicher Darstellung der Programmvorgänge.
    """
    
    # Farbdefinitionen für verschiedene Log-Level
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def __init__(
        self,
        name: str = "trading_console",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None
    ):
        """
        Initialisiert den ConsoleLogger.
        
        Args:
            name: Name des Loggers
            log_level: Logging-Level (DEBUG, INFO, etc.)
            log_file: Optional - Pfad zur Log-Datei
        """
        colorama.init()  # Initialisiere Farbunterstützung
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Formatter für die Konsole mit Farben
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        
        # Custom Filter für farbige Ausgabe
        console_handler.addFilter(self.ColoredOutputFilter())
        
        self.logger.addHandler(console_handler)
        
        # Optional: File Handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            
            self.logger.addHandler(file_handler)
    
    class ColoredOutputFilter(logging.Filter):
        def filter(self, record):
            if record.levelname in ConsoleLogger.COLORS:
                color = ConsoleLogger.COLORS[record.levelname]
                record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
            return True
    
    def debug(self, msg: str, *args, **kwargs):
        """Debug-Level Logging"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Info-Level Logging"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Warning-Level Logging"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Error-Level Logging"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Critical-Level Logging"""
        self.logger.critical(msg, *args, **kwargs)
    
    def section(self, title: str):
        """
        Erstellt einen hervorgehobenen Abschnitt im Log.
        
        Args:
            title: Titel des Abschnitts
        """
        width = 80
        separator = "=" * width
        self.info(separator)
        self.info(f"=== {title} ===".center(width))
        self.info(separator)
    
    def progress(self, current: int, total: int, prefix: str = ""):
        """
        Zeigt einen Fortschrittsbalken an.
        
        Args:
            current: Aktueller Fortschritt
            total: Gesamtanzahl
            prefix: Präfix-Text
        """
        percent = (current / total) * 100
        bar_length = 50
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        self.info(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})")
    
    def success(self, msg: str, *args, **kwargs):
        """Erfolgs-Nachricht mit grüner Hervorhebung"""
        self.info(f"✓ {msg}", *args, **kwargs)
    
    def failure(self, msg: str, *args, **kwargs):
        """Fehler-Nachricht mit roter Hervorhebung"""
        self.error(f"✗ {msg}", *args, **kwargs)
    
    def timing(self, start_time: datetime):
        """
        Zeigt die verstrichene Zeit an.
        
        Args:
            start_time: Startzeitpunkt
        """
        elapsed = datetime.now() - start_time
        self.info(f"Ausführungszeit: {elapsed}") 