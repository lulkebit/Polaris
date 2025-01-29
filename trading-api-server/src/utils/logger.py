import os
import sys
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def setup_logger(name: str):
    """
    Konfiguriert den Logger für das Modul
    """
    log_path = os.getenv('LOG_PATH', 'logs/api_server.log')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Stelle sicher, dass das Logs-Verzeichnis existiert
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Entferne Standard-Handler
    logger.remove()
    
    # Füge Handler für Konsolenausgabe hinzu
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Füge Handler für Datei-Logging hinzu
    logger.add(
        log_path,
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level
    )
    
    return logger.bind(name=name) 