import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from utils.logger import logger
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    Hole Daten für mehrere Aktien parallel und kombiniere sie in einem Dictionary
    """
    all_data = {}
    total_symbols = len(symbols)
    successful_fetches = 0
    failed_fetches = 0
    
    def fetch_with_retry(symbol):
        return symbol, fetch_stock_data(symbol, interval, period)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(fetch_with_retry, symbol): symbol for symbol in symbols}
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol, df = future.result()
                if df is not None and not df.empty:
                    all_data[symbol] = df
                    successful_fetches += 1
                    logger.info(f"Daten für {symbol} erfolgreich geholt: {len(df)} Datenpunkte")
                else:
                    failed_fetches += 1
                    logger.warning(f"Keine Daten für {symbol} verfügbar")
            except Exception as e:
                failed_fetches += 1
                logger.error(f"Fehler beim Abrufen von {symbol}: {str(e)}")
    
    # Zusammenfassung am Ende
    logger.info(f"Abruf abgeschlossen: {successful_fetches} erfolgreich, {failed_fetches} fehlgeschlagen")
    if failed_fetches > 0:
        logger.warning(f"{failed_fetches} Symbole konnten nicht abgerufen werden.")
    
    return all_data

# Beispielaufruf
if __name__ == "__main__":
    try:
        # Hole Daten für mehrere Aktien
        symbols = ["AAPL", "MSFT", "GOOGL"]
        # Verwende korrekte Parameter
        data = fetch_multiple_stocks(symbols, interval="1d", period="1mo")
        for symbol, df in data.items():
            logger.info(f"Testaufruf für {symbol}: {len(df)} Datenpunkte über {len(df.index.unique())} Tage")
    except Exception as e:
        logger.error(f"Testaufruf fehlgeschlagen: {str(e)}")