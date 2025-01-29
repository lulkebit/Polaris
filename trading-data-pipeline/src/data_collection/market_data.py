import os
import requests
import pandas as pd
from dotenv import load_dotenv
from utils.logger import logger
import time
from datetime import datetime, timedelta

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

def fetch_stock_data(symbol: str, interval: str = "5min", adjusted: bool = True) -> pd.DataFrame:
    """
    Hole Aktiendaten von Alpha Vantage.
    Unterstützt verschiedene Intervalle: '1min', '5min', '15min', '30min', '60min', 'daily'
    """
    logger.info(f"Starte Abruf von Aktiendaten für Symbol {symbol} mit Interval {interval}")
    
    # Wähle die passende API-Funktion basierend auf dem Interval
    if interval == 'daily':
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        time_series_key = "Time Series (Daily)"
    else:
        function = "TIME_SERIES_INTRADAY"
        time_series_key = f"Time Series ({interval})"
    
    all_data = []
    
    try:
        # Hole die maximale Datenmenge
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full"  # Hole maximale Datenmenge (bis zu 20 Jahre für tägliche Daten)
        }
        
        if interval != 'daily':
            params["interval"] = interval
        
        logger.info(f"Rufe vollständige historische Daten für {symbol} ab...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Verarbeite die Rohdaten
        time_series = data.get(time_series_key, {})
        if not time_series:
            logger.error(f"Keine Daten für Symbol {symbol} gefunden")
            raise ValueError(f"Keine Daten für Symbol {symbol} gefunden")
        
        df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df.columns = [col.split(" ")[1] for col in df.columns]
        
        # Sortiere nach Datum (neueste zuerst)
        df = df.sort_index(ascending=False)
        
        logger.info(f"Erfolgreich {len(df)} Datenpunkte für {symbol} abgerufen")
        logger.info(f"Zeitraum: von {df.index.min()} bis {df.index.max()}")
        
        # Wenn wir Intraday-Daten holen, warte 12 Sekunden wegen API-Limits
        if interval != 'daily':
            logger.info("Warte 12 Sekunden wegen API-Ratenlimit...")
            time.sleep(12)
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim API-Aufruf für {symbol}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Abruf von {symbol}: {str(e)}")
        raise

def fetch_multiple_stocks(symbols: list, interval: str = "daily") -> dict:
    """
    Hole Daten für mehrere Aktien und kombiniere sie in einem Dictionary
    """
    all_data = {}
    total_symbols = len(symbols)
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"Verarbeite Symbol {i}/{total_symbols}: {symbol}")
            df = fetch_stock_data(symbol, interval, adjusted=True)
            all_data[symbol] = df
            logger.info(f"Daten für {symbol} erfolgreich geholt: {len(df)} Datenpunkte")
            
            # Warte zwischen den API-Aufrufen um das Limit nicht zu überschreiten
            if i < total_symbols:  # Nicht warten nach dem letzten Symbol
                logger.info("Warte 12 Sekunden vor dem nächsten API-Aufruf...")
                time.sleep(12)
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von {symbol}: {str(e)}")
            continue
    
    return all_data

# Beispielaufruf
if __name__ == "__main__":
    try:
        # Hole Daten für mehrere Aktien
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = fetch_multiple_stocks(symbols)
        for symbol, df in data.items():
            logger.info(f"Testaufruf für {symbol}: {len(df)} Datenpunkte über {len(df.index.unique())} Tage")
    except Exception as e:
        logger.error(f"Testaufruf fehlgeschlagen: {str(e)}")