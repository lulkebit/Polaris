import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Union, Optional
import torch
import platform
import psutil

class AILogger:
    """
    Spezialisierter Logger für KI-Trading-Analysen mit strukturierter Ausgabe
    und verschiedenen Log-Levels für unterschiedliche Aspekte der KI-Analyse.
    """
    
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    def __init__(
        self,
        name: str = "trading_ai",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Hauptlogger erstellen
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Formatierung für Console und File
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File Handler für allgemeines Logging
        general_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(general_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Spezialisierte JSON-Logger für verschiedene Aspekte
        self.prediction_log = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.indicators_log = self.log_dir / f"indicators_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.trades_log = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.model_metrics_log = self.log_dir / f"model_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Initial Hardware-Info loggen
        self.log_hardware_info()
    
    def log_hardware_info(self) -> None:
        """
        Erkennt und protokolliert verfügbare und genutzte Hardware-Ressourcen (CPU/GPU).
        """
        hardware_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': 'cpu'
        }
        
        if hardware_info['gpu_available']:
            hardware_info['gpu_devices'] = []
            for i in range(hardware_info['gpu_device_count']):
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                }
                hardware_info['gpu_devices'].append(gpu_info)
            
            # Aktuell genutztes Gerät ermitteln
            hardware_info['current_device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self._log_json(self.model_metrics_log, {
            'model_name': 'system',
            'metrics': {'hardware_info': hardware_info}
        })
        
        # Logging der Hardware-Informationen
        self.logger.info("=== Hardware-Konfiguration ===")
        self.logger.info(f"Platform: {hardware_info['platform']}")
        self.logger.info(f"CPU: {hardware_info['processor']}")
        self.logger.info(f"CPU Kerne: {hardware_info['cpu_cores']} (physisch) / {hardware_info['cpu_threads']} (logisch)")
        self.logger.info(f"RAM: {hardware_info['ram_total_gb']} GB")
        
        if hardware_info['gpu_available']:
            for i, gpu in enumerate(hardware_info['gpu_devices']):
                self.logger.info(f"GPU {i}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
            self.logger.info(f"Aktives Gerät: {hardware_info['current_device']}")
        else:
            self.logger.info("Keine GPU verfügbar - CPU wird verwendet")
        self.logger.info("============================")
    
    def _log_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Speichert strukturierte Daten im JSONL-Format"""
        data['timestamp'] = datetime.now().isoformat()
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    
    def log_prediction(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Protokolliert eine Modellvorhersage mit zugehörigen Details.
        
        Args:
            symbol: Das Handelssymbol
            prediction: Die Vorhersage (z.B. Preisänderung)
            confidence: Konfidenzwert der Vorhersage
            features: Verwendete Features für die Vorhersage
            metadata: Zusätzliche Metadaten
        """
        data = {
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'features': features or {},
            'metadata': metadata or {}
        }
        self._log_json(self.prediction_log, data)
        self.logger.info(f"Prediction for {symbol}: {prediction:.4f} (conf: {confidence:.2f})")
    
    def log_indicators(
        self,
        symbol: str,
        indicators: Dict[str, float],
        timeframe: str = "1d"
    ) -> None:
        """
        Protokolliert technische Indikatoren.
        
        Args:
            symbol: Das Handelssymbol
            indicators: Dict mit Indikatorwerten
            timeframe: Zeitrahmen der Indikatoren
        """
        data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'indicators': indicators
        }
        self._log_json(self.indicators_log, data)
        self.logger.debug(f"Indicators updated for {symbol} ({timeframe})")
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float,
        reason: str,
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Protokolliert Handelsentscheidungen und -ausführungen.
        
        Args:
            symbol: Das Handelssymbol
            action: Art der Handelsaktion (buy/sell)
            price: Ausführungspreis
            size: Handelsgröße
            reason: Grund für den Handel
            risk_metrics: Risikometriken
        """
        data = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'size': size,
            'reason': reason,
            'risk_metrics': risk_metrics or {}
        }
        self._log_json(self.trades_log, data)
        self.logger.info(
            f"Trade: {action.upper()} {symbol} | Price: {price:.2f} | Size: {size:.4f} | Reason: {reason}"
        )
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Protokolliert Modellmetriken mit Zeitstempel."""
        metrics_str = ' | '.join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items())
        data = {
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.now().strftime(self.TIME_FORMAT),
            'metrics_str': metrics_str
        }
        self._log_json(self.model_metrics_log, data)
        self.logger.info(f"Model {model_name} | {metrics_str}")
    
    def get_predictions_df(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Lädt Vorhersagen als DataFrame"""
        return pd.read_json(self.prediction_log, lines=True)
    
    def get_indicators_df(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Lädt Indikatoren als DataFrame"""
        return pd.read_json(self.indicators_log, lines=True)
    
    def get_trades_df(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Lädt Trades als DataFrame"""
        return pd.read_json(self.trades_log, lines=True)
    
    def get_model_metrics_df(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Lädt Modellmetriken als DataFrame"""
        return pd.read_json(self.model_metrics_log, lines=True) 