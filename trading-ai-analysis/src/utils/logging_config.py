import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, Any

class CustomJsonFormatter(logging.Formatter):
    """JSON-Formatter für strukturiertes Logging"""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

class TradeLogger:
    """Zentraler Logger für das Trading-System"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Hauptlogger konfigurieren
        self.logger = logging.getLogger("trading_ai")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Handler konfigurieren
        self._setup_handlers()
    
    def _setup_handlers(self):
        # Formatierung für Konsole
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s'
        )
        
        # Konsolen-Handler (INFO und höher)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # JSON File Handler (alle Logs)
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading_ai.json",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(CustomJsonFormatter())
        
        # Täglicher File Handler für wichtige Ereignisse
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / "daily_trading.log",
            when="midnight",
            interval=1,
            backupCount=30
        )
        daily_handler.setLevel(logging.INFO)
        daily_handler.setFormatter(console_formatter)
        
        # Handler hinzufügen
        self.logger.addHandler(console_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(daily_handler)
    
    def log_trade_analysis(self, analysis_data: Dict[str, Any], level: int = logging.INFO):
        """Spezielles Logging für Handelsanalysen"""
        self.logger.log(level, "Trade Analysis", extra={"extra_data": analysis_data})
    
    def log_strategy_update(self, strategy_data: Dict[str, Any], level: int = logging.INFO):
        """Spezielles Logging für Strategieaktualisierungen"""
        self.logger.log(level, "Strategy Update", extra={"extra_data": strategy_data})
    
    def log_performance_metrics(self, metrics: Dict[str, Any], level: int = logging.INFO):
        """Spezielles Logging für Performance-Metriken"""
        self.logger.log(level, "Performance Metrics", extra={"extra_data": metrics})

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

# Globale Logger-Instanz
trade_logger: Optional[TradeLogger] = None

def setup_logging(log_dir: str = "logs") -> TradeLogger:
    """Initialisiert den globalen Logger"""
    global trade_logger
    if trade_logger is None:
        trade_logger = TradeLogger(log_dir)
    return trade_logger

def get_logger() -> TradeLogger:
    """Gibt die globale Logger-Instanz zurück"""
    if trade_logger is None:
        return setup_logging()
    return trade_logger 