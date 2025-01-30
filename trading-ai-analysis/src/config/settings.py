from pathlib import Path
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Basis-Verzeichnisse
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_HISTORY_DIR = BASE_DIR / "analysis_history"

# Stelle sicher, dass alle benötigten Verzeichnisse existieren
for directory in [DATA_DIR, LOGS_DIR, MODEL_CACHE_DIR, RESULTS_DIR, ANALYSIS_HISTORY_DIR]:
    directory.mkdir(exist_ok=True)

# Datenbank-Konfiguration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "trading_ai"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Analyse-Konfiguration
ANALYSIS_CONFIG = {
    "default_lookback_days": 365,
    "min_data_points": 100,
    "max_optimization_iterations": 1000,
    "risk_free_rate": 0.02,
    "benchmark_symbol": "^GSPC",  # S&P 500
}

# Performance-Modi
PERFORMANCE_MODES: Dict[str, Dict[str, Any]] = {
    "ultra-low": {
        "batch_size": 32,
        "max_workers": 2,
        "use_gpu": False,
    },
    "low": {
        "batch_size": 64,
        "max_workers": 4,
        "use_gpu": False,
    },
    "normal": {
        "batch_size": 128,
        "max_workers": 8,
        "use_gpu": True,
    },
    "high": {
        "batch_size": 256,
        "max_workers": 16,
        "use_gpu": True,
    }
}

# Logging-Konfiguration
LOGGING_CONFIG = {
    "json_log_file": "trading_ai.json",
    "daily_log_file": "daily_trading.log",
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "daily_backup_count": 30,
}

# Modell-Konfiguration
MODEL_CONFIG = {
    "cache_ttl": 24 * 60 * 60,  # 24 Stunden in Sekunden
    "max_cache_size": 1024 * 1024 * 1024,  # 1GB
    "model_version": "v1.0.0",
}

# Trading-Konfiguration
TRADING_CONFIG = {
    "max_position_size": 0.1,  # 10% des Portfolios
    "stop_loss_percentage": 0.02,  # 2%
    "take_profit_percentage": 0.05,  # 5%
    "max_trades_per_day": 10,
    "min_trade_interval": 300,  # 5 Minuten in Sekunden
}

# API-Konfiguration
API_CONFIG = {
    "rate_limit": 60,  # Anfragen pro Minute
    "timeout": 30,  # Sekunden
    "retry_attempts": 3,
    "retry_delay": 5,  # Sekunden
} 