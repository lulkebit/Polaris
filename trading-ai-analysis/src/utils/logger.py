import logging
import os
from datetime import datetime

# ANSI Escape Codes für Farben
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Grün
    'WARNING': '\033[33m',    # Gelb
    'ERROR': '\033[31m',      # Rot
    'CRITICAL': '\033[41m',   # Roter Hintergrund
    'MODEL': '\033[35m',      # Magenta für Modell-bezogene Logs
    'ANALYSIS': '\033[36m',   # Cyan für Analyse-bezogene Logs
    'DATABASE': '\033[34m',   # Blau für Datenbank-bezogene Logs
    'RESET': '\033[0m'        # Reset
}

class ColoredFormatter(logging.Formatter):
    """
    Custom Formatter für farbige Log-Ausgaben mit KI-spezifischen Anpassungen
    """
    def format(self, record):
        # Speichere die originale levelname
        orig_levelname = record.levelname
        
        # Spezielle Formatierung für KI-bezogene Logs
        if hasattr(record, 'model_info'):
            record.msg = f"{COLORS['MODEL']}[MODEL] {record.msg}{COLORS['RESET']}"
        elif hasattr(record, 'analysis_info'):
            record.msg = f"{COLORS['ANALYSIS']}[ANALYSIS] {record.msg}{COLORS['RESET']}"
        elif hasattr(record, 'db_info'):
            record.msg = f"{COLORS['DATABASE']}[DB] {record.msg}{COLORS['RESET']}"
        
        # Standard-Farbformatierung
        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        record.asctime = f"\033[34m{self.formatTime(record)}{COLORS['RESET']}"
        
        # Fehler-Hervorhebung
        if record.levelno >= logging.ERROR:
            record.msg = f"{COLORS['ERROR']}{record.msg}{COLORS['RESET']}"
        
        result = super().format(record)
        record.levelname = orig_levelname
        return result

def setup_logger(name):
    """
    Konfiguriert und erstellt einen Logger mit dem angegebenen Namen.
    Enthält spezielle Anpassungen für KI-Analyse-Logging.
    
    Args:
        name: Name des Loggers (üblicherweise __name__)
        
    Returns:
        logging.Logger: Konfigurierter Logger
    """
    # Erstelle Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Debug-Level für detailliertere KI-Logs
    
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
    
    # File Handler
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'ai_analysis_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Debug-Level für detaillierte Logs in Datei
    file_handler.setFormatter(file_formatter)
    
    # Entferne existierende Handler
    if logger.handlers:
        logger.handlers.clear()
    
    # Füge Handler zum Logger hinzu
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Füge Extra-Methoden für KI-spezifisches Logging hinzu
    def model_log(self, msg, *args, **kwargs):
        kwargs['extra'] = {'model_info': True}
        self.info(msg, *args, **kwargs)
    
    def analysis_log(self, msg, *args, **kwargs):
        kwargs['extra'] = {'analysis_info': True}
        self.info(msg, *args, **kwargs)
    
    def db_log(self, msg, *args, **kwargs):
        kwargs['extra'] = {'db_info': True}
        self.info(msg, *args, **kwargs)
    
    # Füge die neuen Methoden zum Logger hinzu
    logger.model_log = model_log.__get__(logger)
    logger.analysis_log = analysis_log.__get__(logger)
    logger.db_log = db_log.__get__(logger)
    
    return logger 