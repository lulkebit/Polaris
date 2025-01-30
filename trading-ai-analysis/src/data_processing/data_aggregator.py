from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv
from utils.logger import setup_logger
from datetime import datetime, timedelta
from typing import Optional

logger = setup_logger(__name__)

def get_database_connection():
    """Erstellt eine Datenbankverbindung mit den Umgebungsvariablen"""
    load_dotenv()
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    return create_engine(connection_string)

def create_combined_market_data():
    """
    Erstellt und aktualisiert die market_data_combined Tabelle aus den einzelnen Symbol-Tabellen
    """
    logger.info("Starte Aggregation der Marktdaten")
    engine = get_database_connection()
    
    try:
        # Liste aller Aktien-Symbole
        symbols = ["aapl", "msft", "googl", "amzn", "meta", "nvda", "tsla", "jpm", "v", "wmt"]
        all_data = []
        
        # Hole Daten aus jeder Symbol-Tabelle
        for symbol in symbols:
            try:
                table_name = f"market_data_{symbol}"
                query = f"""
                    SELECT 
                        timestamp as date,
                        '{symbol.upper()}' as symbol,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        close_normalized
                    FROM {table_name}
                """
                df = pd.read_sql(query, engine)
                all_data.append(df)
                logger.info(f"Daten für {symbol.upper()} erfolgreich geladen")
            except Exception as e:
                logger.warning(f"Konnte keine Daten für {symbol} laden: {str(e)}")
                continue
        
        if not all_data:
            logger.error("Keine Daten zum Kombinieren gefunden")
            return
        
        # Kombiniere alle Daten
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sortiere nach Datum und Symbol
        combined_df = combined_df.sort_values(['date', 'symbol'])
        
        # Speichere in der kombinierten Tabelle
        combined_df.to_sql(
            'market_data_combined',
            engine,
            if_exists='replace',
            index=False
        )
        
        logger.info(f"Kombinierte Markttabelle erfolgreich erstellt mit {len(combined_df)} Einträgen")
        
    except Exception as e:
        logger.error(f"Fehler bei der Datenaggregation: {str(e)}")
        raise 

class DataAggregator:
    def __init__(self):
        self.engine = get_database_connection()
        self.logger = logger

    def get_market_data(self, days: int = 365) -> pd.DataFrame:
        """Holt Marktdaten für den angegebenen Zeitraum"""
        self.logger.info(f"Hole Marktdaten für die letzten {days} Tage")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = f"""
                SELECT *
                FROM market_data_combined
                WHERE date >= '{start_date.date()}'
                AND date <= '{end_date.date()}'
                ORDER BY date ASC, symbol ASC
            """
            
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Erfolgreich {len(df)} Marktdatensätze geladen")
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktdaten: {str(e)}")
            return pd.DataFrame()  # Leerer DataFrame im Fehlerfall

    def get_news_data(self, days: int = 365) -> pd.DataFrame:
        """Holt Nachrichtendaten für den angegebenen Zeitraum"""
        self.logger.info(f"Hole Nachrichtendaten für die letzten {days} Tage")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = f"""
                SELECT timestamp, title, description, "publishedAt", url, sentiment
                FROM news
                WHERE "publishedAt" >= '{start_date.date()}'
                AND "publishedAt" <= '{end_date.date()}'
                ORDER BY "publishedAt" DESC
            """
            
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Erfolgreich {len(df)} Nachrichtendatensätze geladen")
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Nachrichtendaten: {str(e)}")
            return pd.DataFrame()  # Leerer DataFrame im Fehlerfall 