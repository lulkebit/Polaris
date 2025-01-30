import argparse
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

from config.settings import (
    PERFORMANCE_MODES,
    ANALYSIS_CONFIG,
    TRADING_CONFIG,
    RESULTS_DIR,
    ANALYSIS_HISTORY_DIR
)
from utils.logging_config import setup_logging, get_logger
from database.connection import DatabaseConnection
from analysis.market_analyzer import MarketAnalyzer
from data_processing.data_aggregator import DataAggregator
from models.analysis_pipeline import AnalysisPipeline
from database.trade_repository import TradeRepository
from analysis.risk_manager import RiskManager

class TradingAIAnalysis:
    """Hauptklasse für die Trading-AI-Analyse"""
    
    def __init__(self, performance_mode: str = "normal"):
        self.logger = get_logger()
        self.performance_config = PERFORMANCE_MODES.get(performance_mode, PERFORMANCE_MODES["normal"])
        self.db_connection = DatabaseConnection()
        self.trade_repository = TradeRepository(self.db_connection)
        self.risk_manager = RiskManager(TRADING_CONFIG)
        
        self.logger.logger.info(
            "Trading AI Analysis initialisiert",
            extra={"performance_mode": performance_mode}
        )
    
    def run_analysis(self, mode: str = "new") -> bool:
        """Führt eine neue Analyse durch oder verwendet die letzte Analyse"""
        try:
            if mode == "new":
                return self._run_new_analysis()
            else:
                return self._use_last_analysis()
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler während der Analyse: {str(e)}",
                exc_info=True,
                extra={"error_type": type(e).__name__}
            )
            return False
    
    def _run_new_analysis(self) -> bool:
        """Führt eine neue Analyse durch"""
        try:
            # Initialisiere Pipeline-Komponenten
            data_aggregator = DataAggregator()
            market_analyzer = MarketAnalyzer(history_dir=str(ANALYSIS_HISTORY_DIR))
            analysis_pipeline = AnalysisPipeline(
                data_aggregator=data_aggregator,
                market_analyzer=market_analyzer,
                risk_manager=self.risk_manager,
                config=ANALYSIS_CONFIG
            )
            
            # Führe Pipeline aus
            analysis_results = analysis_pipeline.execute()
            
            # Speichere Handelssignale in der Datenbank
            if analysis_results.trade_signals:
                self.trade_repository.save_trade_signals(analysis_results.trade_signals)
                self.logger.info(
                    "Handelssignale gespeichert",
                    extra={
                        "num_signals": len(analysis_results.trade_signals),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Logge Pipeline-Ergebnisse
            self.logger.info(
                "Analyse-Pipeline Ergebnisse",
                extra={
                    "num_signals": len(analysis_results.trade_signals) if analysis_results.trade_signals else 0,
                    "pipeline_steps": analysis_results.pipeline_steps,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            analysis_results.metrics = {
                'performance': analysis_results.performance_metrics,
                'risk': analysis_results.risk_metrics
            }
            # Speichere Analyseergebnisse
            self._save_analysis_results(analysis_results)
            
            return True
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler während der neuen Analyse: {str(e)}",
                exc_info=True
            )
            return False
    
    def _use_last_analysis(self) -> bool:
        """Verwendet die letzte gespeicherte Analyse aus der Datenbank"""
        try:
            last_signals = self.trade_repository.get_latest_trade_signals()
            if not last_signals:
                self.logger.logger.warning("Keine vorherige Analyse gefunden. Führe neue Analyse durch...")
                return self._run_new_analysis()
                
            self.logger.logger.info(
                "Verwende letzte Analyse",
                extra={"num_signals": len(last_signals)}
            )
            return True
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der letzten Analyse: {str(e)}",
                exc_info=True
            )
            return False

    def _save_analysis_results(self, analysis_results) -> None:
        """Speichert die Analyseergebnisse"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / timestamp
        output_dir.mkdir(exist_ok=True)
        
        # Speichere Analyseergebnisse als JSON
        analysis_results.save_to_json(output_dir / "analysis_results.json")
        
        # Speichere Visualisierungen
        analysis_results.create_visualizations(output_dir)
        
        self.logger.logger.info(
            f"Analyseergebnisse gespeichert",
            extra={"output_dir": str(output_dir)}
        )
    
    def cleanup(self) -> None:
        """Räumt Ressourcen auf"""
        try:
            self.db_connection.close()
        except Exception as e:
            self.logger.logger.error(f"Fehler beim Aufräumen: {str(e)}", exc_info=True)

def main() -> int:
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Trading AI Analysis System")
    parser.add_argument(
        "--performance",
        choices=list(PERFORMANCE_MODES.keys()),
        default="normal",
        help="Performance-Modus für die Analyse"
    )
    parser.add_argument(
        "--mode",
        choices=["new", "last"],
        default="new",
        help="Analysemodus: 'new' für neue Analyse, 'last' für letzte gespeicherte Analyse"
    )
    args = parser.parse_args()
    
    # Initialisiere Logging
    setup_logging()
    logger = get_logger()
    
    try:
        analysis_system = TradingAIAnalysis(performance_mode=args.performance)
        success = analysis_system.run_analysis(mode=args.mode)
        analysis_system.cleanup()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.logger.info("Analyse durch Benutzer unterbrochen")
        return 130
    except Exception as e:
        logger.logger.critical(f"Kritischer Fehler: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 