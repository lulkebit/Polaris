from sqlalchemy import create_engine, text, Table, Column, Integer, Float, DateTime, MetaData, String
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from utils.logger import logger

# Lade Umgebungsvariablen
load_dotenv()

# Erstelle Datenbankverbindung
def get_database_connection():
    """Erstellt eine Datenbankverbindung mit den Umgebungsvariablen"""
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    return create_engine(connection_string)

# Definiere das Schema für die kombinierte Markttabelle
metadata = MetaData()

market_data_combined = Table(
    'market_data_combined',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('timestamp', DateTime, nullable=False),
    Column('symbol', String(10), nullable=False),
    Column('open', Float, nullable=False),
    Column('high', Float, nullable=False),
    Column('low', Float, nullable=False),
    Column('close', Float, nullable=False),
    Column('volume', Float, nullable=False),
    Column('close_normalized', Float),
)

def create_tables():
    """Erstellt die Tabellen in der Datenbank"""
    engine = get_database_connection()
    metadata.create_all(engine)
    logger.info("Tabellen erfolgreich erstellt")

def normalize_price(price: float, min_price: float, max_price: float) -> float:
    """Normalisiert einen Preis auf einen Wert zwischen 0 und 1"""
    if max_price == min_price:
        return 0.5
    return (price - min_price) / (max_price - min_price)

def aggregate_market_data(symbols: List[str], days: int = 365) -> None:
    """
    Aggregiert Marktdaten aus den einzelnen Symbol-Tabellen in die kombinierte Tabelle
    """
    engine = get_database_connection()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Lösche alte Daten
        with engine.connect() as connection:
            connection.execute(text("TRUNCATE TABLE market_data_combined"))
            connection.commit()
        
        all_data = []
        
        # Hole Daten für jedes Symbol
        for symbol in symbols:
            try:
                table_name = f"market_data_{symbol.lower()}"
                query = f"""
                    SELECT 
                        timestamp,
                        '{symbol.upper()}' as symbol,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM {table_name}
                    WHERE timestamp >= :start_date
                    ORDER BY timestamp
                """
                
                df = pd.read_sql(
                    text(query),
                    engine,
                    params={'start_date': start_date}
                )
                
                if not df.empty:
                    # Berechne normalisierte Schlusskurse
                    symbol_min = df['close'].min()
                    symbol_max = df['close'].max()
                    df['close_normalized'] = df['close'].apply(
                        lambda x: normalize_price(x, symbol_min, symbol_max)
                    )
                    
                    all_data.append(df)
                    logger.info(f"Daten für {symbol} erfolgreich geladen")
                
            except Exception as e:
                logger.error(f"Fehler beim Laden der Daten für {symbol}: {str(e)}")
                continue
        
        if not all_data:
            logger.warning("Keine Daten zum Kombinieren gefunden")
            return
        
        # Kombiniere alle Daten
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Speichere in der kombinierten Tabelle
        combined_df.to_sql(
            'market_data_combined',
            engine,
            if_exists='append',
            index=False
        )
        
        logger.info(f"Kombinierte Markttabelle erfolgreich aktualisiert mit {len(combined_df)} Einträgen")
        
    except Exception as e:
        logger.error(f"Fehler bei der Datenaggregation: {str(e)}")
        raise

if __name__ == "__main__":
    # Liste der zu aggregierenden Symbole
    symbols = ["aapl", "msft", "googl", "amzn", "meta", "nvda", "tsla", "jpm", "v", "wmt"]
    
    # Erstelle Tabellen falls sie nicht existieren
    create_tables()
    
    # Aggregiere Daten
    aggregate_market_data(symbols) 