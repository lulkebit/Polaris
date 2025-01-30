from models.deepseek_model import DeepseekAnalyzer
from utils.logger import setup_logger
from data_processing.data_aggregator import create_combined_market_data
import schedule
import time
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Logger einrichten
logger = setup_logger(__name__)

# Performance-Modus aus Umgebungsvariable lesen
PERFORMANCE_MODE = os.getenv('PERFORMANCE_MODE', 'normal')
if PERFORMANCE_MODE == 'low':
    logger.info("Low-Performance-Modus aktiv - Reduzierte Datenmenge und Analysefrequenz")

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

def fetch_latest_data():
    """Holt die neuesten Daten aus der Datenbank"""
    try:
        # Aktualisiere die kombinierte Markttabelle
        create_combined_market_data()
        
        engine = get_database_connection()
        
        # Zeitfenster basierend auf Performance-Modus
        days_window = 2 if PERFORMANCE_MODE == 'low' else 7
        
        # Hole die neuesten Marktdaten mit reduzierter Auflösung im Low-Performance-Modus
        if PERFORMANCE_MODE == 'low':
            market_data = pd.read_sql(f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (date_trunc('hour', date)) *
                    FROM market_data_combined
                    WHERE date >= current_date - interval '{days_window} days'
                ) subq
                ORDER BY date DESC
            """, engine)
        else:
            market_data = pd.read_sql(f"""
                SELECT * FROM market_data_combined
                WHERE date >= current_date - interval '{days_window} days'
                ORDER BY date DESC
            """, engine)
        
        # Hole die neuesten Nachrichten mit reduzierter Menge im Low-Performance-Modus
        news_limit = 50 if PERFORMANCE_MODE == 'low' else 200
        news_data = pd.read_sql(f"""
            SELECT 
                title,
                description as content,
                url,
                "publishedAt" as published_at
            FROM news
            WHERE "publishedAt" >= current_date - interval '{days_window} days'
            ORDER BY "publishedAt" DESC
            LIMIT {news_limit}
        """, engine)
        
        return market_data, news_data
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {str(e)}")
        raise e

def run_analysis():
    """Führt die KI-Analyse durch"""
    try:
        logger.info("Starte KI-Analyse")
        
        # Hole die neuesten Daten
        market_data, news_data = fetch_latest_data()
        logger.info(f"Daten geladen: {len(market_data)} Marktdatensätze, {len(news_data)} Nachrichtendatensätze")
        
        # Initialisiere Analyzer
        analyzer = DeepseekAnalyzer()
        
        # Führe Analyse durch
        analysis_result = analyzer.get_combined_analysis(
            market_data.to_json(),
            news_data.to_json()
        )
        
        # Speichere Ergebnisse
        engine = get_database_connection()
        analysis_data = pd.DataFrame({
            'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')],
            'analysis': [analysis_result]
        })
        analysis_data.to_sql('ai_analysis', engine, if_exists='append', index=False)
        
        logger.info("KI-Analyse erfolgreich abgeschlossen und gespeichert")
        
    except Exception as e:
        logger.error(f"Fehler während der Analyse: {str(e)}")
        raise e

if __name__ == "__main__":
    logger.info("Starte KI-Analyse-System")
    
    # Führe erste Analyse sofort durch
    logger.info("Führe initiale Analyse durch...")
    run_analysis()
    
    # Plane regelmäßige Analysen basierend auf Performance-Modus
    if PERFORMANCE_MODE == 'low':
        logger.info("Konfiguriere 4-stündliche Analysen (Low-Performance-Modus)")
        schedule.every(4).hours.do(run_analysis)
    else:
        logger.info("Konfiguriere stündliche Analysen")
        schedule.every().hour.do(run_analysis)
    
    logger.info("Analyse-Scheduler gestartet")
    while True:
        schedule.run_pending()
        time.sleep(60) 