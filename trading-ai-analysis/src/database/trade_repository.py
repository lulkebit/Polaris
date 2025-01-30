from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from database.connection import DatabaseConnection
from database.schema import TradeSignal, TradingAIAnalysis
from utils.logging_config import get_logger

class TradeRepository:
    """Repository für die Verwaltung von Handelssignalen"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.logger = get_logger()
    
    def save_trade_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Speichert Handelssignale in der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                for signal in signals:
                    trade_signal = TradeSignal(
                        symbol=signal["symbol"],
                        direction=signal["direction"],
                        entry_price=signal["entry_price"],
                        stop_loss=signal.get("stop_loss"),
                        take_profit=signal.get("take_profit"),
                        confidence=signal.get("confidence", 0.0),
                        signal_time=signal.get("signal_time", datetime.now()),
                        analysis_id=signal.get("analysis_id"),
                        metadata=signal.get("metadata", {})
                    )
                    session.add(trade_signal)
                session.commit()
                
                self.logger.log_trade_analysis({
                    "action": "save_signals",
                    "count": len(signals),
                    "timestamp": datetime.now().isoformat()
                })
                return True
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Speichern der Handelssignale: {str(e)}",
                exc_info=True
            )
            return False
    
    def get_pending_signals(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Holt ausstehende Handelssignale aus der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                query = session.query(TradeSignal)\
                    .filter(TradeSignal.status == "pending")\
                    .order_by(TradeSignal.signal_time.desc())
                
                if limit:
                    query = query.limit(limit)
                
                signals = query.all()
                return [signal.to_dict() for signal in signals]
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der ausstehenden Signale: {str(e)}",
                exc_info=True
            )
            return []
    
    def update_signal_status(self, signal_id: int, new_status: str, metadata: Optional[Dict] = None) -> bool:
        """Aktualisiert den Status eines Handelssignals"""
        try:
            with self.db_connection.get_session() as session:
                signal = session.query(TradeSignal).get(signal_id)
                if signal:
                    signal.status = new_status
                    signal.updated_at = datetime.now()
                    if metadata:
                        signal.metadata.update(metadata)
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Aktualisieren des Signalstatus: {str(e)}",
                exc_info=True
            )
            return False
    
    def get_signals_by_analysis(self, analysis_id: int) -> List[Dict[str, Any]]:
        """Holt alle Handelssignale für eine bestimmte Analyse"""
        try:
            with self.db_connection.get_session() as session:
                signals = session.query(TradeSignal)\
                    .filter(TradeSignal.analysis_id == analysis_id)\
                    .order_by(TradeSignal.signal_time.desc())\
                    .all()
                return [signal.to_dict() for signal in signals]
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der Signale für Analyse {analysis_id}: {str(e)}",
                exc_info=True
            )
            return []
    
    def cleanup_old_signals(self, days: int = 30) -> int:
        """Bereinigt alte Handelssignale"""
        try:
            with self.db_connection.get_session() as session:
                cutoff_date = datetime.now() - datetime.timedelta(days=days)
                deleted = session.query(TradeSignal)\
                    .filter(TradeSignal.signal_time < cutoff_date)\
                    .delete()
                session.commit()
                return deleted
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Bereinigen alter Signale: {str(e)}",
                exc_info=True
            )
            return 0 