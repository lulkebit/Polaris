from data_collection.market_data import fetch_multiple_stocks
from data_collection.news_data import fetch_news
from data_processing.clean_market_data import clean_market_data
from data_processing.news_processor import add_sentiment
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
    try:
        logger.info("Starte Pipeline-Durchlauf")
        
        # Initialisiere Datenbankverbindung
        db = DatabaseConnection()
        
        # 1. Hole Marktdaten für alle Aktien
        logger.info(f"Hole Marktdaten für {len(STOCK_SYMBOLS)} Aktien")
        market_data_dict = fetch_multiple_stocks(STOCK_SYMBOLS, interval="daily")
        
        # Verarbeite jede Aktie
        for symbol, market_df in market_data_dict.items():
            try:
                logger.info(f"Verarbeite Daten für {symbol}")
                
                # Bereinige Daten
                cleaned_market_df = clean_market_data(market_df)
                logger.info(f"Marktdaten für {symbol} erfolgreich bereinigt")
                
                # Speichere in eigener Tabelle
                table_name = f"market_data_{symbol.lower()}"
                db.save_to_database(cleaned_market_df, table_name)
                logger.info(f"Marktdaten für {symbol} in Tabelle {table_name} gespeichert")
                
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von {symbol}: {str(e)}")
                continue
        
        # 2. Hole und verarbeite Nachrichten
        logger.info("Hole Nachrichtendaten")
        news_df = fetch_news()
        logger.info(f"Erfolgreich {len(news_df)} Nachrichtendatensätze geholt")
        
        news_with_sentiment = add_sentiment(news_df)
        logger.info("Sentiment-Analyse für Nachrichten abgeschlossen")
        
        db.save_to_database(news_with_sentiment, "news")
        logger.info("Nachrichtendaten in Datenbank gespeichert")
        
        logger.info("Pipeline-Durchlauf erfolgreich abgeschlossen")
    except Exception as e:
        logger.error(f"Fehler während des Pipeline-Durchlaufs: {str(e)}")
        raise e

if __name__ == "__main__":
    logger.info("Starte Trading Data Pipeline")
    
    # Führe die Pipeline sofort einmal aus
    logger.info("Führe initiale Pipeline aus...")
    run_pipeline()
    
    # Plane zukünftige Ausführungen
    logger.info("Konfiguriere tägliche Ausführung")
    schedule.every().day.at("00:00").do(run_pipeline)  # Führe täglich um Mitternacht aus
    
    logger.info("Pipeline-Scheduler gestartet")
    while True:
        schedule.run_pending()
        time.sleep(1)