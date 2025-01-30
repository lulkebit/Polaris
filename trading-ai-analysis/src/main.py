import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from database.connection import DatabaseConnection
from database.schema import MarketAnalysis, BacktestResult, OptimizationResult, TradingAIAnalysis
from analysis.market_analyzer import MarketAnalyzer
from backtesting.visualizer import BacktestVisualizer, BacktestResults
from optimization.strategy_optimizer import StrategyOptimizer
from data_processing.data_aggregator import DataAggregator

def setup_logging():
    """Konfiguriert das Logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setze Root-Logger auf WARNING, um doppelte Logs zu vermeiden
    logging.getLogger().setLevel(logging.WARNING)
    
    # Konfiguriere nur den App-Logger
    app_logger = logging.getLogger("trading_ai")
    app_logger.setLevel(logging.INFO)
    
    # Formatierung
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_dir / "trading_ai.log")
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)
    
    # Verhindere Weiterleitung an Parent-Logger
    app_logger.propagate = False

def run_new_analysis(db_connection: DatabaseConnection):
    """Führt eine neue Analyse durch"""
    logger = logging.getLogger(__name__)
    logger.info("Starte neue Analyse...")
    
    try:
        # Initialisiere Komponenten
        market_analyzer = MarketAnalyzer()
        strategy_optimizer = StrategyOptimizer()
        
        # Lade Marktdaten (Beispiel: die letzten 365 Tage)
        data_aggregator = DataAggregator()
        market_data = data_aggregator.get_market_data(days=365)
        news_data = data_aggregator.get_news_data(days=365)
        
        # Führe Marktanalyse durch
        analysis_results = market_analyzer.analyze_data(market_data=market_data, news_data=news_data)
        
        # Optimiere Strategie basierend auf Analyseergebnissen
        optimization_results, backtest_results_raw = strategy_optimizer.optimize_strategy(
            market_data=market_data,
            news_data=news_data
        )
        
        # Konvertiere in BacktestResults Objekt
        backtest_results = BacktestResults(
            equity_curve=pd.Series(backtest_results_raw.equity_curve) if hasattr(backtest_results_raw, 'equity_curve') else pd.Series(),
            trades=backtest_results_raw.trades if hasattr(backtest_results_raw, 'trades') else [],
            monthly_returns=pd.Series(backtest_results_raw.monthly_returns) if hasattr(backtest_results_raw, 'monthly_returns') else pd.Series(),
            performance_summary=backtest_results_raw.performance_metrics if hasattr(backtest_results_raw, 'performance_metrics') else {},
            risk_metrics=backtest_results_raw.risk_metrics if hasattr(backtest_results_raw, 'risk_metrics') else {},
            position_history=backtest_results_raw.position_history if hasattr(backtest_results_raw, 'position_history') else [],
            benchmark_comparison=backtest_results_raw.benchmark_comparison if hasattr(backtest_results_raw, 'benchmark_comparison') else {}
        )
        
        # Erstelle Visualizer und generiere Visualisierungen
        backtest_visualizer = BacktestVisualizer(backtest_results)
        
        # Erstelle Ausgabeverzeichnis falls nicht vorhanden
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generiere Visualisierungen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_visualizer.create_performance_dashboard(str(output_dir / f"performance_dashboard_{timestamp}.html"))
        backtest_visualizer.plot_trade_analysis(str(output_dir / f"trade_analysis_{timestamp}.html"))
        backtest_visualizer.create_risk_report(str(output_dir / f"risk_report_{timestamp}.html"))
        
        # Speichere Ergebnisse in der Datenbank
        with db_connection.get_session() as session:
            session.add_all([
                *analysis_results,
                *optimization_results,
                *backtest_results_raw
            ])
        
        logger.info(f"Neue Analyse erfolgreich abgeschlossen. Visualisierungen in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Fehler während der Analyse: {str(e)}")
        raise

def load_existing_analysis(db_connection: DatabaseConnection):
    """Lädt und visualisiert vorhandene Analysedaten"""
    logger = logging.getLogger(__name__)
    logger.info("Lade vorhandene Analysedaten...")
    
    try:
        with db_connection.get_session() as session:
            # Lade die neuesten Ergebnisse
            latest_analysis = session.query(TradingAIAnalysis)\
                .order_by(TradingAIAnalysis.timestamp.desc())\
                .first()
                
            latest_backtest = session.query(BacktestResult)\
                .order_by(BacktestResult.end_time.desc())\
                .first()
                
            latest_optimization = session.query(OptimizationResult)\
                .order_by(OptimizationResult.end_time.desc())\
                .first()
        
        if not latest_analysis:
            logger.warning("Keine vollständigen Analysedaten in der Datenbank gefunden")
            return
            
        # Konvertiere die Datenbank-Ergebnisse in BacktestResults
        if latest_backtest:
            backtest_results = BacktestResults(
                equity_curve=pd.Series(latest_backtest.equity_curve) if hasattr(latest_backtest, 'equity_curve') else pd.Series(),
                trades=latest_backtest.trades if hasattr(latest_backtest, 'trades') else [],
                monthly_returns=pd.Series(latest_backtest.monthly_returns) if hasattr(latest_backtest, 'monthly_returns') else pd.Series(),
                performance_summary=latest_backtest.performance_metrics if hasattr(latest_backtest, 'performance_metrics') else {},
                risk_metrics=latest_backtest.risk_metrics if hasattr(latest_backtest, 'risk_metrics') else {},
                position_history=latest_backtest.position_history if hasattr(latest_backtest, 'position_history') else [],
                benchmark_comparison=latest_backtest.benchmark_comparison if hasattr(latest_backtest, 'benchmark_comparison') else {}
            )
            
            # Erstelle Visualizer mit den konvertierten Daten
            backtest_visualizer = BacktestVisualizer(backtest_results)
            
            # Erstelle Ausgabeverzeichnis falls nicht vorhanden
            output_dir = Path("analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            # Generiere Visualisierungen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backtest_visualizer.create_performance_dashboard(str(output_dir / f"performance_dashboard_{timestamp}.html"))
            backtest_visualizer.plot_trade_analysis(str(output_dir / f"trade_analysis_{timestamp}.html"))
            backtest_visualizer.create_risk_report(str(output_dir / f"risk_report_{timestamp}.html"))
            
            logger.info(f"Vorhandene Analysedaten erfolgreich geladen und visualisiert. Ausgabe in: {output_dir}")
        
        # Optimiere Strategien basierend auf den vorhandenen Daten
        if latest_analysis and latest_analysis.market_data_analysis:
            strategy_optimizer = StrategyOptimizer()
            optimization_results = strategy_optimizer.optimize_strategies(latest_analysis.market_data_analysis)
            
            # Speichere neue Optimierungsergebnisse
            if optimization_results:
                with db_connection.get_session() as session:
                    session.add(optimization_results)
                    session.commit()
                
                logger.info("Strategieoptimierung basierend auf vorhandenen Daten abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der Analysedaten: {str(e)}")
        raise

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Trading AI Analysis System")
    parser.add_argument("--mode", choices=['new', 'existing'], default='new',
                      help="Analysemodus: 'new' für neue Analyse, 'existing' für vorhandene Daten")
    parser.add_argument("--performance", choices=['normal', 'low', 'ultra-low', 'auto'],
                      default='normal', help="Performance-Modus für die Analyse")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("trading_ai")  # Verwende den App-Logger
    
    try:
        # Initialisiere Datenbankverbindung
        db_connection = DatabaseConnection()
        db_connection.initialize()
        
        if args.mode == 'new':
            logger.info(f"Starte neue Analyse im {args.performance}-Performance-Modus")
            run_new_analysis(db_connection)
        else:
            logger.info(f"Lade vorhandene Daten im {args.performance}-Performance-Modus")
            load_existing_analysis(db_connection)
            
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {str(e)}")
        raise
    finally:
        db_connection.close()

if __name__ == "__main__":
    main() 