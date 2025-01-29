import logging
import os
from logging.handlers import RotatingFileHandler

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
    
    # Formatierung für Log-Einträge
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File Handler (mit Rotation)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pipeline.log')
    file_handler = RotatingFileHandler(
        log_dir,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Füge Handler zum Logger hinzu (nur wenn sie noch nicht existieren)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Erstelle einen Standard-Logger für das Modul
logger = setup_logger(__name__)