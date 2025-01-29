import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ANSI Escape Codes für Farben
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Grün
    'WARNING': '\033[33m',   # Gelb
    'ERROR': '\033[31m',     # Rot
    'CRITICAL': '\033[41m',  # Roter Hintergrund
    'RESET': '\033[0m'       # Reset
}

class ColoredFormatter(logging.Formatter):
    """
    Custom Formatter für farbige Log-Ausgaben
    """
    def format(self, record):
        # Speichere die originale levelname
        orig_levelname = record.levelname
        # Füge Farben hinzu
        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        
        # Füge Zeitstempel-Farbe hinzu (Blau)
        record.asctime = f"\033[34m{self.formatTime(record)}{COLORS['RESET']}"
        
        # Füge Farbe zur Nachricht hinzu, wenn es sich um einen Fehler handelt
        if record.levelno >= logging.ERROR:
            record.msg = f"{COLORS['ERROR']}{record.msg}{COLORS['RESET']}"
        
        result = super().format(record)
        # Stelle original levelname wieder her
        record.levelname = orig_levelname
        return result

def setup_logger(name):
    """
    Konfiguriert und erstellt einen Logger mit dem angegebenen Namen.
    
    Args:
        name: Name des Loggers (üblicherweise __name__)
        
    Returns:
        logging.Logger: Konfigurierter Logger
    """
    # Erstelle Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formatierung für Console (mit Farben)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formatierung für File (ohne Farben)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File Handler (mit Rotation)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Entferne existierende Handler
    if logger.handlers:
        logger.handlers.clear()
    
    # Füge Handler zum Logger hinzu
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Erstelle einen Standard-Logger für das Modul
logger = setup_logger(__name__)