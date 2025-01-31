from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
from typing import Optional

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
                logger.info(f"Verbunden mit {self.name} auf {self.host}:{self.port}")
            return engine
        except Exception as e:
            logger.error(f"Verbindungsfehler: {str(e)}")
            raise

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
            SELECT * FROM market_data 
            WHERE date = (SELECT MAX(date) FROM market_data)
        """
        try:
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Fehler beim Laden der neuesten Daten: {str(e)}")
            raise 