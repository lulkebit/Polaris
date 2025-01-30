import os
import sys
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import Json
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from contextlib import contextmanager
from dotenv import load_dotenv
import configparser
from sqlalchemy import create_engine, text, Column, String, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import time

# Füge den src-Ordner zum Python-Path hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

Base = declarative_base()

class AIAnalysis(Base):
    __tablename__ = 'ai_analysis'
    
    timestamp = Column(DateTime, primary_key=True)
    analysis = Column(JSON(none_as_null=True))  # PostgreSQL JSONB Typ für effiziente JSON-Speicherung

class TradingAIAnalysis(Base):
    __tablename__ = 'trading_ai_analyses'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    analysis_type = Column(String(50), nullable=False)
    market_data_analysis = Column(JSON(none_as_null=True), nullable=False)
    news_analysis = Column(JSON(none_as_null=True), nullable=False)
    position_analysis = Column(JSON(none_as_null=True), nullable=False)
    historical_analysis = Column(JSON(none_as_null=True), nullable=False)
    recommendations = Column(JSON(none_as_null=True), nullable=False)
    confidence_score = Column(Integer, nullable=False)

class DatabaseManager:
    def __init__(self):
        self.db_config = self._load_db_config()
        self._initialize_database()
        
        # Erstelle Engine und Session
        self.engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
        # Erstelle Tabellen
        Base.metadata.create_all(self.engine)
        
        # Erstelle Session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info("Datenbank-Manager erfolgreich initialisiert")

    def _load_db_config(self) -> Dict[str, str]:
        """Lädt die Datenbankkonfiguration aus den Umgebungsvariablen"""
        try:
            # Lade .env Datei
            load_dotenv()
            
            return {
                'dbname': os.getenv('DB_NAME', 'polaris'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres'),
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432')
            }
        except Exception as e:
            logger.error(f"Fehler beim Laden der Datenbankkonfiguration: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """Stellt eine Datenbankverbindung her"""
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Initialisiert die Datenbankstruktur"""
        try:
            start_time = time.time()
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Backtesting-Ergebnisse
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_backtest_results'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_backtest_results (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        strategy_name VARCHAR(255),
                        total_return FLOAT,
                        sharpe_ratio FLOAT,
                        max_drawdown FLOAT,
                        win_rate FLOAT,
                        profit_factor FLOAT,
                        num_trades INTEGER,
                        strategy_config JSONB,
                        performance_metrics JSONB,
                        risk_metrics JSONB
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_backtest_results' beschrieben mit: id, timestamp, strategy_name, total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, num_trades, strategy_config, performance_metrics, risk_metrics (Dauer: {time.time() - table_start:.2f}s)")

                # Einzelne Trades
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_trades'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_trades (
                        id SERIAL PRIMARY KEY,
                        backtest_id INTEGER REFERENCES trading_ai_backtest_results(id),
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20),
                        action VARCHAR(20),
                        quantity FLOAT,
                        price FLOAT,
                        pnl FLOAT,
                        commission FLOAT,
                        slippage FLOAT
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_trades' beschrieben mit: id, backtest_id (FK), timestamp, symbol, action, quantity, price, pnl, commission, slippage (Dauer: {time.time() - table_start:.2f}s)")

                # KI-Analysen
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_analyses'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_analyses (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        analysis_type VARCHAR(50),
                        market_data_analysis JSONB,
                        news_analysis JSONB,
                        position_analysis JSONB,
                        historical_analysis JSONB,
                        recommendations JSONB,
                        confidence_score INTEGER
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_analyses' beschrieben mit: id, timestamp, analysis_type, market_data_analysis, news_analysis, position_analysis, historical_analysis, recommendations, confidence_score (Dauer: {time.time() - table_start:.2f}s)")

                # Strategieoptimierungen
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_strategy_optimizations'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_strategy_optimizations (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        iteration INTEGER,
                        strategy_config JSONB,
                        performance_metrics JSONB,
                        improvement_suggestions JSONB,
                        is_best_strategy BOOLEAN
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_strategy_optimizations' beschrieben mit: id, timestamp, iteration, strategy_config, performance_metrics, improvement_suggestions, is_best_strategy (Dauer: {time.time() - table_start:.2f}s)")

                # Portfolio-Snapshots
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_portfolio_snapshots'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_portfolio_snapshots (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        total_value FLOAT,
                        cash_position FLOAT,
                        positions JSONB,
                        sector_allocation JSONB,
                        risk_metrics JSONB
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_portfolio_snapshots' beschrieben mit: id, timestamp, total_value, cash_position, positions, sector_allocation, risk_metrics (Dauer: {time.time() - table_start:.2f}s)")

                # Risikomanagement-Ereignisse
                table_start = time.time()
                logger.info("Beschreibe Tabelle 'trading_ai_risk_events'...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_ai_risk_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        event_type VARCHAR(50),
                        description TEXT,
                        affected_positions JSONB,
                        action_taken TEXT,
                        impact_value FLOAT
                    )
                """)
                logger.debug(f"Tabelle 'trading_ai_risk_events' beschrieben mit: id, timestamp, event_type, description, affected_positions, action_taken, impact_value (Dauer: {time.time() - table_start:.2f}s)")

                conn.commit()
                logger.info(f"Datenbankstruktur erfolgreich initialisiert (Gesamtdauer: {time.time() - start_time:.2f}s)")

        except Exception as e:
            logger.error(f"Fehler bei der Datenbankinitialisierung: {str(e)}")
            raise

    def save_backtest_results(self, results: Dict[str, Any]) -> int:
        """Speichert Backtesting-Ergebnisse"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Hauptergebnisse speichern
                cursor.execute("""
                    INSERT INTO trading_ai_backtest_results (
                        timestamp,
                        strategy_name,
                        total_return,
                        sharpe_ratio,
                        max_drawdown,
                        win_rate,
                        profit_factor,
                        num_trades,
                        strategy_config,
                        performance_metrics,
                        risk_metrics
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    datetime.now(),
                    results.get('strategy_name', 'unnamed'),
                    results['total_return'],
                    results['sharpe_ratio'],
                    results['max_drawdown'],
                    results['win_rate'],
                    results['profit_factor'],
                    len(results['trades']),
                    Json(results.get('strategy_config', {})),
                    Json(results['performance_summary']),
                    Json(results['risk_metrics'])
                ))
                
                backtest_id = cursor.fetchone()[0]
                
                # Trades speichern
                self._save_trades(cursor, backtest_id, results['trades'])
                
                conn.commit()
                return backtest_id
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Backtest-Ergebnisse: {str(e)}")
            raise

    def save_ai_analysis(self, data: dict) -> None:
        """Speichert eine AI-Analyse in der Datenbank"""
        try:
            # Stelle sicher, dass die Analyse als JSON-String vorliegt
            analysis_data = data['analysis']
            if not isinstance(analysis_data, str):
                analysis_data = json.dumps(analysis_data)
            
            # Parse den JSON-String um sicherzustellen, dass er valide ist
            analysis_json = json.loads(analysis_data)
            
            # Erstelle neuen Analyseeintrag
            analysis = AIAnalysis(
                timestamp=datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S'),
                analysis=analysis_json
            )
            
            # Füge zur Session hinzu und committe
            self.session.add(analysis)
            self.session.commit()
            logger.info("AI-Analyse erfolgreich in Datenbank gespeichert")
            
        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim JSON-Parsing der Analyse: {str(e)}")
            self.session.rollback()
            raise
        except Exception as e:
            logger.error(f"Fehler beim Speichern der AI-Analyse: {str(e)}")
            self.session.rollback()
            raise

    def save_optimization_result(self, optimization: Dict[str, Any]) -> int:
        """Speichert Strategieoptimierungen"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_ai_strategy_optimizations (
                        timestamp,
                        iteration,
                        strategy_config,
                        performance_metrics,
                        improvement_suggestions,
                        is_best_strategy
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    datetime.now(),
                    optimization['iteration'],
                    Json(optimization['strategy_config']),
                    Json(optimization['performance_metrics']),
                    Json(optimization.get('improvement_suggestions', [])),
                    optimization.get('is_best_strategy', False)
                ))
                
                optimization_id = cursor.fetchone()[0]
                conn.commit()
                return optimization_id
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Optimierung: {str(e)}")
            raise

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """Speichert Portfolio-Snapshots"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_ai_portfolio_snapshots (
                        timestamp,
                        total_value,
                        cash_position,
                        positions,
                        sector_allocation,
                        risk_metrics
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    datetime.now(),
                    snapshot['total_value'],
                    snapshot['cash_position'],
                    Json(snapshot['positions']),
                    Json(snapshot.get('sector_allocation', {})),
                    Json(snapshot.get('risk_metrics', {}))
                ))
                
                snapshot_id = cursor.fetchone()[0]
                conn.commit()
                return snapshot_id
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Portfolio-Snapshots: {str(e)}")
            raise

    def save_risk_event(self, event: Dict[str, Any]) -> int:
        """Speichert Risikomanagement-Ereignisse"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_ai_risk_events (
                        timestamp,
                        event_type,
                        description,
                        affected_positions,
                        action_taken,
                        impact_value
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    datetime.now(),
                    event['event_type'],
                    event['description'],
                    Json(event.get('affected_positions', [])),
                    event.get('action_taken', ''),
                    event.get('impact_value', 0.0)
                ))
                
                event_id = cursor.fetchone()[0]
                conn.commit()
                return event_id
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Risikoereignisses: {str(e)}")
            raise

    def _save_trades(self, cursor, backtest_id: int, trades: List[Dict[str, Any]]) -> None:
        """Hilfsmethode zum Speichern von Trades"""
        for trade in trades:
            cursor.execute("""
                INSERT INTO trading_ai_trades (
                    backtest_id,
                    timestamp,
                    symbol,
                    action,
                    quantity,
                    price,
                    pnl,
                    commission,
                    slippage
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                backtest_id,
                trade['timestamp'],
                trade['symbol'],
                trade['action'],
                trade['quantity'],
                trade['price'],
                trade.get('pnl', 0.0),
                trade.get('commission', 0.0),
                trade.get('slippage', 0.0)
            ))

    def get_backtest_results(self, limit: int = 10) -> pd.DataFrame:
        """Lädt die letzten Backtest-Ergebnisse"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trading_ai_backtest_results 
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_optimization_history(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Lädt die Optimierungshistorie"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trading_ai_strategy_optimizations 
                WHERE strategy_name = %s OR %s IS NULL
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=(strategy_name, strategy_name))

    def get_portfolio_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Lädt die Portfolio-Historie für einen Zeitraum"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trading_ai_portfolio_snapshots 
                WHERE timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            return pd.read_sql_query(query, conn, params=(start_date, end_date))

    def get_risk_events(self, event_type: Optional[str] = None) -> pd.DataFrame:
        """Lädt Risikomanagement-Ereignisse"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trading_ai_risk_events 
                WHERE event_type = %s OR %s IS NULL
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=(event_type, event_type))

    def get_trades_for_backtest(self, backtest_id: int) -> pd.DataFrame:
        """Lädt alle Trades für einen bestimmten Backtest"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trading_ai_trades 
                WHERE backtest_id = %s
                ORDER BY timestamp
            """
            return pd.read_sql_query(query, conn, params=(backtest_id,))

    def get_latest_analyses(self, limit: int = 10) -> list:
        """Holt die neuesten Analysen aus der Datenbank"""
        try:
            analyses = self.session.query(TradingAIAnalysis)\
                .order_by(TradingAIAnalysis.timestamp.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    'timestamp': analysis.timestamp.isoformat(),
                    'analysis_type': analysis.analysis_type,
                    'market_data_analysis': analysis.market_data_analysis,
                    'news_analysis': analysis.news_analysis,
                    'position_analysis': analysis.position_analysis,
                    'historical_analysis': analysis.historical_analysis,
                    'recommendations': analysis.recommendations,
                    'confidence_score': analysis.confidence_score
                }
                for analysis in analyses
            ]
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der neuesten Analysen: {str(e)}")
            raise

    def get_analyses_by_timerange(self, start_date: datetime, end_date: datetime) -> list:
        """Holt Analysen in einem bestimmten Zeitraum"""
        try:
            analyses = self.session.query(TradingAIAnalysis)\
                .filter(TradingAIAnalysis.timestamp.between(start_date, end_date))\
                .order_by(TradingAIAnalysis.timestamp.desc())\
                .all()
            
            return [
                {
                    'timestamp': analysis.timestamp.isoformat(),
                    'analysis_type': analysis.analysis_type,
                    'market_data_analysis': analysis.market_data_analysis,
                    'news_analysis': analysis.news_analysis,
                    'position_analysis': analysis.position_analysis,
                    'historical_analysis': analysis.historical_analysis,
                    'recommendations': analysis.recommendations,
                    'confidence_score': analysis.confidence_score
                }
                for analysis in analyses
            ]
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Analysen nach Zeitraum: {str(e)}")
            raise

    @contextmanager
    def get_session(self):
        """Stellt eine Datenbank-Session zur Verfügung"""
        session = Session(bind=self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def __del__(self):
        """Cleanup beim Beenden"""
        if hasattr(self, 'session'):
            self.session.close() 