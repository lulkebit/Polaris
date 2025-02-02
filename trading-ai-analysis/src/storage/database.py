from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        load_dotenv()
        
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.name = os.getenv("DB_NAME", "trading_analysis")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")
        
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        
    def _create_engine(self):
        conn_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        try:
            engine = create_engine(
                conn_str,
                pool_size=5,
                max_overflow=10,
                pool_recycle=300
            )
            with engine.connect() as conn:
                self._initialize_tables(conn)
                logger.info(f"Verbunden mit {self.name} auf {self.host}:{self.port}")
            return engine
        except Exception as e:
            logger.error(f"Verbindungsfehler: {str(e)}")
            raise

    def _initialize_tables(self, connection):
        """Initialisiert die Datenbanktabellen"""
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS market_data_combined (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                open DECIMAL(10,2) NOT NULL,
                high DECIMAL(10,2) NOT NULL,
                low DECIMAL(10,2) NOT NULL,
                close DECIMAL(10,2) NOT NULL,
                volume BIGINT NOT NULL,
                close_normalized DECIMAL(10,6),
                analysis_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS news_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                published_at TIMESTAMP NOT NULL,
                url TEXT,
                sentiment DECIMAL(4,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

    def execute_query(self, query: str, params: Optional[dict] = None):
        with self.engine.connect() as connection:
            try:
                result = connection.execute(text(query), params or {})
                if result.returns_rows:
                    return result.fetchall()
                return result.rowcount
            except Exception as e:
                connection.rollback()
                logger.error(f"Query fehlgeschlagen: {str(e)}")
                raise

    def get_session(self):
        return self.Session()

    def get_latest_data(self):
        """Lädt die neuesten Handelsdaten aus der Datenbank"""
        query = """
            WITH RankedNews AS (
                SELECT 
                    symbol,
                    published_at,
                    title,
                    description,
                    url,
                    sentiment,
                    ROW_NUMBER() OVER (PARTITION BY symbol, DATE(published_at) ORDER BY published_at DESC) as rn
                FROM news_data
                WHERE published_at >= (SELECT MAX(timestamp) - INTERVAL '30 days' FROM market_data_combined)
            ),
            DailyNews AS (
                SELECT 
                    symbol,
                    DATE(published_at) as news_date,
                    AVG(sentiment) as daily_sentiment,
                    COUNT(*) as daily_news_count,
                    STRING_AGG(
                        CASE WHEN rn <= 5 THEN title ELSE NULL END,
                        ' ||| ' ORDER BY published_at DESC
                    ) as recent_titles,
                    STRING_AGG(
                        CASE WHEN rn <= 5 THEN description ELSE NULL END,
                        ' ||| ' ORDER BY published_at DESC
                    ) as recent_descriptions,
                    STRING_AGG(
                        CASE WHEN rn <= 5 THEN url ELSE NULL END,
                        ' ||| ' ORDER BY published_at DESC
                    ) as recent_urls,
                    STRING_AGG(
                        CASE WHEN rn <= 5 THEN CAST(sentiment AS TEXT) ELSE NULL END,
                        ' ||| ' ORDER BY published_at DESC
                    ) as recent_sentiments
                FROM RankedNews
                GROUP BY symbol, DATE(published_at)
            )
            SELECT 
                m.symbol, 
                m.timestamp as date, 
                m.open, 
                m.high, 
                m.low, 
                m.close, 
                m.volume,
                m.close_normalized,
                n.recent_titles as news_titles,
                n.recent_descriptions as news_descriptions,
                n.recent_urls as news_urls,
                n.recent_sentiments as news_sentiments,
                n.daily_sentiment as avg_sentiment,
                n.daily_news_count as news_count
            FROM market_data_combined m
            LEFT JOIN DailyNews n 
                ON m.symbol = n.symbol 
                AND DATE(m.timestamp) = n.news_date
            WHERE m.timestamp >= (SELECT MAX(timestamp) - INTERVAL '30 days' FROM market_data_combined)
            ORDER BY m.timestamp DESC, n.daily_sentiment DESC
        """
        try:
            result = self.execute_query(query)
            return pd.DataFrame(result, columns=[
                'symbol', 'date', 'open', 'high', 'low', 'close', 
                'volume', 'close_normalized', 'news_titles', 'news_descriptions',
                'news_urls', 'news_sentiments', 'avg_sentiment', 'news_count'
            ])
        except Exception as e:
            logger.error(f"Fehler beim Laden der neuesten Daten: {str(e)}")
            raise 