import os
import requests
import pandas as pd
from dotenv import load_dotenv
from utils.logger import logger
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
MAX_RETRIES = 3
RETRY_DELAY = 20  # Sekunden zwischen Retry-Versuchen
API_CALL_DELAY = 1  # Sekunden zwischen API-Aufrufen
_api_limit_reached = None  # Cache für API-Limit-Status

def check_api_limit() -> bool:
    """
    Prüft, ob das API-Limit erreicht wurde, indem ein einfacher API-Aufruf gemacht wird
    Returns:
        bool: True wenn das Limit erreicht wurde, False wenn noch Aufrufe möglich sind
    """
    global _api_limit_reached
    
    # Wenn wir den Status schon geprüft haben, verwende den gecachten Wert
    if _api_limit_reached is not None:
        return _api_limit_reached
    
    logger.info("Prüfe API-Limit Status...")
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": "IBM",  # Nutze IBM als Test-Symbol
            "interval": "1min",
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Prüfe auf API-Limit-Meldungen
        if "Note" in data and "API call frequency" in data["Note"]:
            logger.warning("API-Tageslimit erreicht. Weitere Abfragen werden übersprungen.")
            _api_limit_reached = True
            return True
        
        logger.info("API-Limit nicht erreicht, Abfragen können fortgesetzt werden.")
        _api_limit_reached = False
        return False
        
    except Exception as e:
        logger.error(f"Fehler beim Prüfen des API-Limits: {str(e)}")
        _api_limit_reached = True
        return True  # Im Zweifelsfall annehmen, dass das Limit erreicht ist

def check_api_response(data: Dict[str, Any]) -> None:
    """
    Überprüft die API-Antwort auf Fehlermeldungen oder API-Limits
    """
    if "Error Message" in data:
        raise ValueError(f"API-Fehler: {data['Error Message']}")
    if "Note" in data:
        if "API call frequency" in data["Note"]:
            global _api_limit_reached
            _api_limit_reached = True
            raise ValueError("API-Limit erreicht. Bitte warten Sie einen Moment.")
        logger.warning(f"API-Hinweis: {data['Note']}")

def fetch_stock_data(symbol: str, interval: str = "5min", adjusted: bool = True) -> Optional[pd.DataFrame]:
    """
    Hole Aktiendaten von Alpha Vantage mit Retry-Logik.
    Unterstützt verschiedene Intervalle: '1min', '5min', '15min', '30min', '60min', 'daily'
    """
    # Prüfe zuerst das API-Limit
    if check_api_limit():
        logger.warning(f"API-Limit erreicht - Überspringe Abruf für Symbol {symbol}")
        return None
        
    logger.info(f"Starte Abruf von Aktiendaten für Symbol {symbol} mit Interval {interval}")
    
    # Wähle die passende API-Funktion basierend auf dem Interval
    if interval == 'daily':
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        time_series_key = "Time Series (Daily)"
    else:
        function = "TIME_SERIES_INTRADAY"
        time_series_key = f"Time Series ({interval})"
    
    for attempt in range(MAX_RETRIES):
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_KEY,
                "outputsize": "full"
            }
            
            if interval != 'daily':
                params["interval"] = interval
            
            logger.info(f"Versuch {attempt + 1}/{MAX_RETRIES}: Rufe Daten für {symbol} ab...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Überprüfe API-Antwort auf Fehler
            check_api_response(data)
            
            # Verarbeite die Rohdaten
            time_series = data.get(time_series_key, {})
            if not time_series:
                logger.error(f"Keine Daten für Symbol {symbol} gefunden")
                return None
            
            df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.columns = [col.split(" ")[1] for col in df.columns]
            
            # Sortiere nach Datum (neueste zuerst)
            df = df.sort_index(ascending=False)
            
            logger.info(f"Erfolgreich {len(df)} Datenpunkte für {symbol} abgerufen")
            logger.info(f"Zeitraum: von {df.index.min()} bis {df.index.max()}")
            
            return df
            
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Fehler beim API-Aufruf für {symbol} (Versuch {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.info(f"Warte {wait_time} Sekunden vor dem nächsten Versuch...")
                time.sleep(wait_time)
            else:
                logger.error(f"Maximale Anzahl von Versuchen für {symbol} erreicht")
                return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abruf von {symbol}: {str(e)}")
            return None

def fetch_multiple_stocks(symbols: list, interval: str = "daily") -> dict:
    """
    Hole Daten für mehrere Aktien und kombiniere sie in einem Dictionary
    """
    # Prüfe zuerst das API-Limit
    if check_api_limit():
        logger.warning("API-Limit erreicht - Überspringe alle Marktdaten-Abfragen")
        return {}
        
    all_data = {}
    total_symbols = len(symbols)
    successful_fetches = 0
    failed_fetches = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"Verarbeite Symbol {i}/{total_symbols}: {symbol}")
            
            # Warte vor jedem API-Aufruf (außer dem ersten)
            if i > 1:
                logger.info(f"Warte {API_CALL_DELAY} Sekunden vor dem nächsten API-Aufruf...")
                time.sleep(API_CALL_DELAY)
            
            df = fetch_stock_data(symbol, interval, adjusted=True)
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
            continue
    
    # Zusammenfassung am Ende
    logger.info(f"Abruf abgeschlossen: {successful_fetches} erfolgreich, {failed_fetches} fehlgeschlagen")
    if failed_fetches > 0:
        logger.warning("Einige Symbole konnten nicht abgerufen werden. Überprüfen Sie das API-Limit oder die Symbolnamen.")
    
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