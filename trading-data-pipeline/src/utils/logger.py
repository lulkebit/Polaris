import logging
import sys

# Erstelle einen Handler für die Dateiausgabe
file_handler = logging.FileHandler("pipeline.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Erstelle einen Handler für die Konsolenausgabe
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Konfiguriere den Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Verhindere die Weitergabe von Logs an übergeordnete Handler
logger.propagate = False