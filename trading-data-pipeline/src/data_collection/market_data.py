import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from utils.logger import logger
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_processing.market_data_aggregator import get_database_connection

load_dotenv()

MAX_RETRIES = 3
RETRY_DELAY = 5  # Sekunden zwischen Retry-Versuchen
MAX_WORKERS = 5  # Maximale Anzahl paralleler Downloads

# Gültige Intervalle und Perioden für die Yahoo Finance API
VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', 'max']

def validate_parameters(interval: str, period: str) -> tuple[str, str]:
    """
    Validiert und korrigiert die Eingabeparameter für die Yahoo Finance API.
    """
    # Konvertiere gängige Intervall-Formate
    interval_mapping = {
        'daily': '1d',
        'weekly': '1wk',
        'monthly': '1mo',
        'minute': '1m',
        'hourly': '1h'
    }
    
    # Normalisiere das Intervall
    normalized_interval = interval_mapping.get(interval, interval)
    
    if normalized_interval not in VALID_INTERVALS:
        logger.warning(f"Ungültiges Intervall '{interval}'. Verwende Standard '1d'.")
        normalized_interval = '1d'
    
    # Validiere die Periode
    if period not in VALID_PERIODS:
        logger.warning(f"Ungültige Periode '{period}'. Verwende Standard '1mo'.")
        period = '1mo'
    
    return normalized_interval, period

def fetch_stock_data(symbol: str, interval: str = "1d", period: str = "1mo") -> Optional[pd.DataFrame]:
    """
    Hole Aktiendaten von Yahoo Finance mit Retry-Logik.
    Unterstützt verschiedene Intervalle: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    Unterstützt verschiedene Perioden: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', 'max'
    """
    # Validiere die Parameter
    interval, period = validate_parameters(interval, period)
    logger.info(f"Starte Abruf von Aktiendaten für Symbol {symbol} mit Interval {interval} und Periode {period}")
    
    for attempt in range(MAX_RETRIES):
        try:
            # Erstelle Ticker-Objekt
            ticker = yf.Ticker(symbol)
            
            # Hole historische Daten
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.error(f"Keine Daten für Symbol {symbol} gefunden")
                return None
            
            # Bereinige die Spalten
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            logger.info(f"Erfolgreich {len(df)} Datenpunkte für {symbol} abgerufen")
            logger.info(f"Zeitraum: von {df.index.min()} bis {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim API-Aufruf für {symbol} (Versuch {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.info(f"Warte {wait_time} Sekunden vor dem nächsten Versuch...")
                time.sleep(wait_time)
            else:
                logger.error(f"Maximale Anzahl von Versuchen für {symbol} erreicht")
                return None

def fetch_multiple_stocks(symbols: list, interval: str = "1d", period: str = "1mo") -> dict:
    """
    Lädt Daten für mehrere Aktien parallel herunter.
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(fetch_stock_data, symbol, interval, period): symbol 
            for symbol in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"Fehler beim Laden der Daten für {symbol}: {str(e)}")
    
    return results

def save_to_database(data: Dict[str, pd.DataFrame]) -> None:
    """
    Speichert die Marktdaten in der Datenbank.
    """
    engine = get_database_connection()
    
    for symbol, df in data.items():
        if df is not None and not df.empty:
            # Bereinige die Spaltennamen
            df = df.rename(columns={
                'Date': 'timestamp',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            table_name = f"market_data_{symbol.lower()}"
            try:
                df.to_sql(table_name, engine, if_exists='append', index=True, index_label='timestamp')
                logger.info(f"Daten für {symbol} erfolgreich in der DB gespeichert")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der Daten für {symbol}: {str(e)}")

def update_market_data():
    """
    Hauptfunktion zum Aktualisieren der Marktdaten.
    """
    symbols = ["aapl", "msft", "googl", "amzn", "meta", "nvda", "tsla", "jpm", "v", "wmt"]
    
    try:
        # Hole neue Daten
        market_data = fetch_multiple_stocks(symbols)
        
        # Speichere in einzelnen Tabellen
        save_to_database(market_data)
        
        # Aggregiere in die kombinierte Tabelle
        from data_processing.market_data_aggregator import aggregate_market_data
        aggregate_market_data(symbols)
        
        logger.info("Marktdaten-Update erfolgreich abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler beim Update der Marktdaten: {str(e)}")
        raise

if __name__ == "__main__":
    update_market_data()