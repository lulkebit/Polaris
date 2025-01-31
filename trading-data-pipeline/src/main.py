from data_collection.market_data import update_market_data
from data_collection.news_data import fetch_news, save_news_to_db
from data_processing.clean_market_data import clean_market_data
from data_processing.news_processor import add_sentiment
from data_processing.market_data_aggregator import create_tables, aggregate_market_data
from storage.database import DatabaseConnection
from utils.logger import logger
import schedule
import time
import pandas as pd

# Liste der zu überwachenden Aktien
STOCK_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Google
    "AMZN",  # Amazon
    "META",  # Meta (Facebook)
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "WMT"    # Walmart
]

def run_pipeline():
    """
    Führt die komplette Daten-Pipeline aus:
    1. Marktdaten sammeln und in Einzeltabellen speichern
    2. News sammeln und Sentiment-Analyse durchführen
    3. Daten in kombinierte Tabelle aggregieren
    """
    try:
        # Stelle sicher, dass alle Tabellen existieren
        create_tables()
        
        # Aktualisiere Marktdaten
        update_market_data()
        
        # Hole und verarbeite News
        for symbol in STOCK_SYMBOLS:
            news_df = fetch_news(f"{symbol} stock")
            processed_news = add_sentiment(news_df)
            save_news_to_db(symbol, processed_news)
            logger.info(f"News für {symbol} verarbeitet und gespeichert")
        
        logger.info("Pipeline erfolgreich ausgeführt")
        
    except Exception as e:
        logger.error(f"Fehler bei der Ausführung der Pipeline: {str(e)}")
        raise

def schedule_pipeline():
    """
    Plant die regelmäßige Ausführung der Pipeline
    - Werktags: Jede Stunde zwischen 8:00 und 22:00 Uhr
    """
    # Plane stündliche Ausführung
    for hour in range(8, 23):
        schedule.every().day.at(f"{hour:02d}:00").do(run_pipeline)
    
    logger.info("Pipeline-Zeitplan erstellt")
    
    while True:
        if time.localtime().tm_wday < 5:  # 0-4 entspricht Montag-Freitag
            schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Führe Pipeline sofort einmal aus
    run_pipeline()
    
    # Starte geplante Ausführung
    schedule_pipeline()