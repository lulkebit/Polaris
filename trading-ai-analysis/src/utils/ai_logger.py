import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import torch
import platform
import psutil
from .console_logger import ConsoleLogger
import traceback

class AILogger(ConsoleLogger):
    """
    Spezialisierter Logger für KI-Trading-Analysen, der den ConsoleLogger erweitert.
    Fokussiert sich auf strukturierte Datenprotokollierung in JSONL-Format.
    """
    
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    def __init__(
        self,
        name: str = "trading_ai",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialisiert den AILogger und richtet die JSONL-Logging-Dateien ein.
        """
        # Debug-Datei für detaillierte Logs
        debug_file = Path(log_dir) / f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        
        # ConsoleLogger-Initialisierung mit Logging-Datei
        log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        super().__init__(
            name=name,
            log_level=console_level,
            log_file=str(log_file),
            debug_file=str(debug_file)
        )
        
        # JSONL-Logging-Dateien Setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        date_suffix = datetime.now().strftime('%Y%m%d')
        self.prediction_log = self.log_dir / f"predictions_{date_suffix}.jsonl"
        self.indicators_log = self.log_dir / f"indicators_{date_suffix}.jsonl"
        self.trades_log = self.log_dir / f"trades_{date_suffix}.jsonl"
        self.model_metrics_log = self.log_dir / f"model_metrics_{date_suffix}.jsonl"
        self.performance_log = self.log_dir / f"performance_{date_suffix}.jsonl"
        self.error_log = self.log_dir / f"errors_{date_suffix}.jsonl"
        self.debug_log = self.log_dir / f"detailed_debug_{date_suffix}.jsonl"

    def _log_json(self, file_path: Path, data: Dict[str, Any], category: str = None) -> None:
        """Speichert strukturierte Daten im JSONL-Format mit zusätzlichen Metadaten"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'data': data,
            'system_info': {
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'cpu_percent': psutil.Process().cpu_percent(),
                'thread_id': psutil.Process().num_threads()
            }
        }
        
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
        
        # Debug-Log für alle JSON-Einträge
        self.debug(f"JSON-Log geschrieben: {file_path.name}", extra={
            'category': category,
            'data_size': len(str(data)),
            'file_path': str(file_path)
        })

    def log_hardware_info(self) -> None:
        """Erkennt und protokolliert Hardware-Ressourcen."""
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
            hardware_info['current_device'] = 'cuda:0'
        
        # Hardware-Info in JSONL speichern
        self._log_json(self.model_metrics_log, {
            'model_name': 'system',
            'metrics': {'hardware_info': hardware_info}
        }, 'system_info')
        
        # Einmalige Konsolenausgabe der Hardware-Konfiguration
        self.section("Hardware-Konfiguration")
        self.info(f"Platform: {hardware_info['platform']}")
        self.info(f"CPU: {hardware_info['processor']}")
        self.info(f"CPU Kerne: {hardware_info['cpu_cores']} (physisch) / {hardware_info['cpu_threads']} (logisch)")
        self.info(f"RAM: {hardware_info['ram_total_gb']} GB")
        
        if hardware_info['gpu_available']:
            for i, gpu in enumerate(hardware_info['gpu_devices']):
                self.info(f"GPU {i}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
        else:
            self.info("Keine GPU verfügbar - CPU wird verwendet")
    
    def log_model_performance(
        self,
        model_name: str,
        batch_size: int,
        input_shape: tuple,
        inference_time: float,
        memory_usage: Dict[str, float],
        device_info: Dict[str, Any]
    ) -> None:
        """Detailliertes Performance-Logging für Modellinferenz"""
        performance_data = {
            'model_name': model_name,
            'batch_size': batch_size,
            'input_shape': input_shape,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'device_info': device_info,
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.performance_log, performance_data, 'model_performance')
        self.debug("Performance-Metriken geloggt", extra=performance_data)

    def log_error_details(
        self,
        error: Exception,
        context: Dict[str, Any],
        stack_trace: str,
        severity: str = "ERROR"
    ) -> None:
        """Detailliertes Error-Logging mit Kontext"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stack_trace': stack_trace,
            'severity': severity,
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.error_log, error_data, 'error')
        self.error(f"Fehler aufgetreten: {str(error)}", error_details=error_data)

    def log_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        phase: str = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Erweiterte Modellmetriken mit Phasen-Tracking"""
        metrics_data = {
            'model_name': model_name,
            'metrics': metrics,
            'phase': phase,
            'additional_info': additional_info or {},
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.model_metrics_log, metrics_data, 'model_metrics')
        
        # Formatierte Debug-Ausgabe
        metrics_str = ' | '.join(
            f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.debug(
            f"Model {model_name} Metrics" + (f" ({phase})" if phase else ""),
            extra={'metrics': metrics, 'phase': phase}
        )

    def log_prediction(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_version: str = None,
        prediction_type: str = None
    ) -> None:
        """Erweiterte Vorhersageprotokollierung mit Modellversion und Typ"""
        prediction_data = {
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'features': features or {},
            'metadata': metadata or {},
            'model_version': model_version,
            'prediction_type': prediction_type,
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.prediction_log, prediction_data, 'prediction')
        self.debug(
            f"Prediction for {symbol}: {prediction:.4f} (conf: {confidence:.2f})",
            extra=prediction_data
        )

    def log_indicators(
        self,
        symbol: str,
        indicators: Dict[str, float],
        timeframe: str = "1d",
        calculation_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Erweiterte Indikatorenprotokollierung mit Berechnungsdetails"""
        indicator_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'indicators': indicators,
            'calculation_details': calculation_details or {},
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.indicators_log, indicator_data, 'indicators')
        self.debug(
            f"Indicators updated for {symbol} ({timeframe})",
            extra=indicator_data
        )

    def log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float,
        reason: str,
        risk_metrics: Optional[Dict[str, float]] = None,
        execution_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Erweiterte Handelsprotokolle mit Ausführungsdetails"""
        trade_data = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'size': size,
            'reason': reason,
            'risk_metrics': risk_metrics or {},
            'execution_details': execution_details or {},
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.trades_log, trade_data, 'trade')
        self.info(
            f"Trade: {action.upper()} {symbol} | Price: {price:.2f} | Size: {size:.4f}",
            category='TRADE'
        )

    def log_debug_info(
        self,
        category: str,
        message: str,
        details: Dict[str, Any],
        stack_info: bool = True
    ) -> None:
        """Detaillierte Debug-Informationen mit Stack-Trace"""
        debug_data = {
            'category': category,
            'message': message,
            'details': details,
            'stack_trace': traceback.extract_stack() if stack_info else None,
            'timestamp': datetime.now().strftime(self.TIME_FORMAT)
        }
        self._log_json(self.debug_log, debug_data, 'debug')
        self.debug(message, extra=details)

    # DataFrame-Zugriffsmethoden mit zusätzlicher Filterung
    def get_predictions_df(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Lädt gefilterte Vorhersagen als DataFrame"""
        df = pd.read_json(self.prediction_log, lines=True)
        return self._filter_dataframe(df, start_date, end_date, symbols)

    def get_indicators_df(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Lädt gefilterte Indikatoren als DataFrame"""
        df = pd.read_json(self.indicators_log, lines=True)
        return self._filter_dataframe(df, start_date, end_date, symbols)

    def get_trades_df(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Lädt gefilterte Trades als DataFrame"""
        df = pd.read_json(self.trades_log, lines=True)
        return self._filter_dataframe(df, start_date, end_date, symbols)

    def get_model_metrics_df(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Lädt gefilterte Modellmetriken als DataFrame"""
        df = pd.read_json(self.model_metrics_log, lines=True)
        if model_names:
            df = df[df['model_name'].isin(model_names)]
        return self._filter_dataframe(df, start_date, end_date)

    def _filter_dataframe(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Hilfsfunktion zum Filtern von DataFrames"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
        
        if symbols and 'symbol' in df.columns:
            df = df[df['symbol'].isin(symbols)]
        
        return df 