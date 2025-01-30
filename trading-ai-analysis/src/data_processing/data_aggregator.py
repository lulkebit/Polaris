from sqlalchemy import create_engine, text, Integer, Float, String, DateTime
from sqlalchemy.sql import or_
import pandas as pd
import os
from dotenv import load_dotenv
from utils.logger import setup_logger
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
import yfinance as yf
from sqlalchemy.orm import Session
import time

from database.connection import DatabaseConnection
from database.schema import MarketData, NewsData
from utils.logging_config import get_logger
from config.settings import API_CONFIG

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
    Erstellt und aktualisiert die market_data_combined Tabelle
    """
    logger.info("Starte Aggregation der Marktdaten")
    engine = get_database_connection()
    
    try:
        query = """
            SELECT 
                id,
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                close_normalized,
                analysis_id
            FROM market_data_combined
            WHERE open IS NOT NULL 
              AND high IS NOT NULL 
              AND low IS NOT NULL 
              AND close IS NOT NULL 
              AND volume IS NOT NULL
            ORDER BY timestamp, symbol
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.error("Keine Daten in der market_data_combined Tabelle gefunden")
            return
        
        # Definiere Datentypen für die Spalten
        dtype_mapping = {
            'id': 'Int64',
            'timestamp': 'datetime64[ns]',
            'symbol': 'string',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'close_normalized': 'float64',
            'analysis_id': 'Int64'
        }
        
        # Konvertiere Datentypen
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        # Fülle fehlende Werte in optionalen Spalten
        df['close_normalized'] = df['close_normalized'].fillna(0.0)
        df['analysis_id'] = df['analysis_id'].fillna(0)
        
        # Speichere in der kombinierten Tabelle mit definierten Datentypen
        df.to_sql(
            'market_data_combined',
            engine,
            if_exists='replace',
            index=False,
            dtype={
                'id': Integer,
                'timestamp': DateTime,
                'symbol': String(20),
                'open': Float,
                'high': Float,
                'low': Float,
                'close': Float,
                'volume': Float,
                'close_normalized': Float,
                'analysis_id': Integer
            }
        )
        
        logger.info(f"Markttabelle erfolgreich aktualisiert mit {len(df)} Einträgen")
        
    except Exception as e:
        logger.error(f"Fehler bei der Datenaggregation: {str(e)}")
        raise

class DataAggregator:
    """Sammelt und aggregiert Markt- und Nachrichtendaten"""
    
    def __init__(self):
        self.logger = get_logger()
        self.db_connection = DatabaseConnection()
        
        # Cache für Daten
        self._market_data_cache = {}
        self._news_data_cache = {}
        self._cache_timeout = 300  # 5 Minuten
    
    def get_market_data(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 365,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Holt Marktdaten für die angegebenen Symbole"""
        try:
            if symbols is None:
                symbols = ["^GSPC"]  # S&P 500 als Standard
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Prüfe Cache
            if use_cache:
                cached_data = self._get_cached_market_data(symbols, start_date)
                if cached_data is not None:
                    return cached_data
            
            # Hole neue Daten
            all_data = []
            for symbol in symbols:
                try:
                    # Versuche zuerst Daten aus der Datenbank zu laden
                    db_data = self._load_market_data_from_db(symbol, start_date)
                    if db_data is not None:
                        all_data.append(db_data)
                        continue
                    
                    # Wenn keine DB-Daten verfügbar, hole von Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if data.empty:
                        self.logger.logger.warning(f"Keine Daten für Symbol {symbol} gefunden")
                        continue
                    
                    # Konvertiere Spaltennamen zu Kleinbuchstaben
                    data.columns = data.columns.str.lower()
                    
                    # Füge Symbol-Spalte hinzu
                    data["symbol"] = symbol
                    
                    # Setze Index als timestamp Spalte
                    data = data.reset_index()
                    if 'datetime' in data.columns:
                        data = data.rename(columns={'datetime': 'timestamp'})
                    elif 'date' in data.columns:
                        data = data.rename(columns={'date': 'timestamp'})
                    else:
                        data['timestamp'] = data.index
                    
                    # Stelle sicher, dass timestamp ein Datetime-Objekt ist
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    
                    # Füge fehlende Spalten hinzu
                    if 'close_normalized' not in data.columns:
                        data['close_normalized'] = 0.0
                    if 'analysis_id' not in data.columns:
                        data['analysis_id'] = 0
                    
                    # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
                    required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    if missing_columns:
                        raise ValueError(f"Fehlende Spalten in den Marktdaten: {missing_columns}")
                    
                    # Entferne Zeilen mit NULL-Werten in wichtigen Spalten
                    data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                    
                    # Fülle optionale Spalten mit Standardwerten
                    data['close_normalized'] = data['close_normalized'].fillna(0.0)
                    data['analysis_id'] = data['analysis_id'].fillna(0)
                    
                    # Speichere in der Datenbank
                    self._save_market_data_to_db(data)
                    
                    all_data.append(data)
                    
                except Exception as e:
                    self.logger.logger.error(
                        f"Fehler beim Laden der Daten für {symbol}: {str(e)}",
                        exc_info=True
                    )
            
            if not all_data:
                raise ValueError("Keine Marktdaten gefunden")
            
            # Kombiniere alle Daten
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Stelle sicher, dass keine NULL-Werte in wichtigen Spalten existieren
            if combined_data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                self.logger.logger.warning("NULL-Werte in Marktdaten gefunden, werden entfernt")
                combined_data = combined_data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Aktualisiere Cache
            self._update_market_data_cache(combined_data, symbols)
            
            return combined_data
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der Marktdaten: {str(e)}",
                exc_info=True
            )
            raise
    
    def get_news_data(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 7,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Holt Nachrichtendaten aus der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                # Hole alle vorhandenen Symbole
                distinct_symbols = session.query(NewsData.symbol).distinct().all()
                available_symbols = [s[0] for s in distinct_symbols]
                
                if not available_symbols:
                    logger.warning("Keine Symbole in der News-Datenbank gefunden")
                    return pd.DataFrame()

                # Falls keine Symbole angegeben, verwende alle vorhandenen
                if symbols is None:
                    symbols = available_symbols
                else:
                    # Filtere nicht vorhandene Symbole heraus
                    symbols = [s for s in symbols if s in available_symbols]
                    if not symbols:
                        logger.warning("Keine der angeforderten Symbole in der News-Datenbank vorhanden")
                        return pd.DataFrame()

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Prüfe Cache
                if use_cache:
                    cached_news = self._get_cached_news_data(symbols, start_date)
                    if cached_news is not None:
                        return cached_news
                
                # Hole Nachrichten aus der Datenbank
                news_query = session.query(NewsData).filter(
                    NewsData.published_at >= start_date,
                    NewsData.published_at <= end_date
                )
                
                # Filtere nach Symbolen wenn angegeben
                news_query = news_query.filter(NewsData.symbol.in_(symbols))
                
                news_data = news_query.all()
                
                if not news_data:
                    self.logger.logger.warning(f"Keine Nachrichtendaten für die angegebenen Symbole gefunden")
                    return pd.DataFrame()
                
                # Konvertiere zu DataFrame
                news_df = pd.DataFrame([
                    {
                        "id": news.id,
                        "symbol": news.symbol,
                        "title": news.title,
                        "description": news.description,
                        "published_at": news.published_at,
                        "url": news.url,
                        "sentiment": news.sentiment,
                        "created_at": news.created_at
                    }
                    for news in news_data
                ])
                
                # Aktualisiere Cache
                self._update_news_data_cache(news_df, symbols)
                
                return news_df
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der Nachrichtendaten: {str(e)}",
                exc_info=True
            )
            return pd.DataFrame()
    
    def _get_cached_market_data(
        self,
        symbols: List[str],
        start_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Holt Marktdaten aus dem Cache"""
        cache_key = f"{','.join(sorted(symbols))}_{start_date.date()}"
        cached = self._market_data_cache.get(cache_key)
        
        if cached is None:
            return None
            
        data, timestamp = cached
        if (datetime.now() - timestamp).total_seconds() > self._cache_timeout:
            # Cache ist abgelaufen
            del self._market_data_cache[cache_key]
            return None
            
        # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_normalized', 'analysis_id']
        if not all(col in data.columns for col in required_columns):
            # Cache ist ungültig
            del self._market_data_cache[cache_key]
            return None
            
        return data
    
    def _update_market_data_cache(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> None:
        """Aktualisiert den Marktdaten-Cache"""
        # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_normalized', 'analysis_id']
        if not all(col in data.columns for col in required_columns):
            logger.warning("Cache-Update übersprungen: Fehlende Spalten in den Daten")
            return
            
        cache_key = f"{','.join(sorted(symbols))}_{datetime.now().date()}"
        self._market_data_cache[cache_key] = (data, datetime.now())
    
    def _get_cached_news_data(
        self,
        symbols: List[str],
        start_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Holt Nachrichtendaten aus dem Cache"""
        cache_key = f"{','.join(sorted(symbols))}_{start_date.date()}"
        cached = self._news_data_cache.get(cache_key)
        
        if cached is None:
            return None
            
        data, timestamp = cached
        if (datetime.now() - timestamp).total_seconds() > self._cache_timeout:
            return None
            
        return data
    
    def _update_news_data_cache(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> None:
        """Aktualisiert den Nachrichtendaten-Cache"""
        cache_key = f"{','.join(sorted(symbols))}_{datetime.now().date()}"
        self._news_data_cache[cache_key] = (data, datetime.now())
    
    def _load_market_data_from_db(
        self,
        symbol: str,
        start_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Lädt Marktdaten aus der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                query = session.query(MarketData)\
                    .filter(
                        MarketData.timestamp >= start_date,
                        MarketData.symbol == symbol.upper()
                    )\
                    .all()
                
                if not query:
                    return None
                
                # Konvertiere zu DataFrame
                data = pd.DataFrame([{
                    'timestamp': row.timestamp,
                    'symbol': row.symbol,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume,
                    'close_normalized': row.close_normalized if row.close_normalized is not None else 0.0,
                    'analysis_id': row.analysis_id if row.analysis_id is not None else 0
                } for row in query])
                
                # Entferne Zeilen mit NULL-Werten in wichtigen Spalten
                data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                
                return data
                
        except Exception as e:
            self.logger.error(
                f"Fehler beim Laden der Marktdaten aus der DB: {str(e)}",
                exc_info=True
            )
            return None
    
    def _save_market_data_to_db(self, data: pd.DataFrame) -> None:
        """Speichert Marktdaten in der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                for _, row in data.iterrows():
                    market_data = MarketData(
                        timestamp=row["timestamp"],
                        symbol=row["symbol"],
                        open=row["open"] if "open" in row else row["Open"],
                        high=row["high"] if "high" in row else row["High"],
                        low=row["low"] if "low" in row else row["Low"],
                        close=row["close"] if "close" in row else row["Close"],
                        volume=row["volume"] if "volume" in row else row["Volume"],
                        close_normalized=row.get("close_normalized"),
                        analysis_id=row.get("analysis_id")
                    )
                    session.add(market_data)
                session.commit()
                
        except Exception as e:
            self.logger.error(
                f"Fehler beim Speichern der Marktdaten in der DB: {str(e)}",
                exc_info=True
            )
    
    def _load_news_from_db(
        self,
        symbol: str,
        start_date: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Lädt Nachrichtendaten aus der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                news = session.query(NewsData)\
                    .filter(
                        NewsData.symbols.contains([symbol]),
                        NewsData.published_at >= start_date
                    )\
                    .all()
                
                if not news:
                    return None
                
                return [
                    {
                        "title": n.title,
                        "content": n.content,
                        "source": n.source,
                        "timestamp": n.published_at,
                        "sentiment_score": n.sentiment_score,
                        "relevance_score": n.relevance_score,
                        "symbols": n.symbols
                    }
                    for n in news
                ]
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Laden der Nachrichten aus der DB: {str(e)}",
                exc_info=True
            )
            return None
    
    def _save_news_to_db(
        self,
        news: List[Dict[str, Any]],
        symbol: str
    ) -> None:
        """Speichert Nachrichtendaten in der Datenbank"""
        try:
            with self.db_connection.get_session() as session:
                for article in news:
                    news_data = NewsData(
                        title=article["title"],
                        content=article.get("content"),
                        source=article["source"],
                        published_at=article["timestamp"],
                        sentiment_score=article.get("sentiment_score"),
                        relevance_score=article.get("relevance_score"),
                        symbols=[symbol]
                    )
                    session.add(news_data)
                session.commit()
                
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Speichern der Nachrichten in der DB: {str(e)}",
                exc_info=True
            )
    
    def _fetch_news_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Holt Nachrichten von der News API"""
        try:
            # Entferne ^ von Indexsymbolen für die Suche
            search_term = symbol.replace("^", "")
            
            response = self.news_api.get_everything(
                q=search_term,
                language="en",
                sort_by="relevancy",
                page_size=100
            )
            
            if not response or "articles" not in response:
                return []
            
            # Konvertiere und bereinige Nachrichten
            news = []
            for article in response["articles"]:
                news.append({
                    "title": article["title"],
                    "content": article.get("content", ""),
                    "source": article["source"]["name"],
                    "timestamp": datetime.strptime(
                        article["publishedAt"],
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "url": article["url"]
                })
            
            return news
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Abrufen der Nachrichten von der API: {str(e)}",
                exc_info=True
            )
            return []

    def _fetch_market_data(self, symbol: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """Holt Marktdaten mit Retry-Logik"""
        max_retries = int(os.getenv("API_RETRY_ATTEMPTS", 3))
        retry_delay = int(os.getenv("API_RETRY_DELAY", 10))
        timeout = int(os.getenv("API_REQUEST_TIMEOUT", 30))
        
        for attempt in range(max_retries):
            try:
                # Prüfe API-Limit
                if not self._check_api_rate_limit():
                    wait_time = self._get_rate_limit_wait_time()
                    logger.info(f"API-Limit erreicht. Warte {wait_time} Sekunden...")
                    time.sleep(wait_time)
                
                # Hole Daten von Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=datetime.now(),
                    timeout=timeout
                )
                
                if data.empty:
                    logger.warning(f"Keine Daten für Symbol {symbol} gefunden (Versuch {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
                    
                logger.info(f"Daten für {symbol} erfolgreich abgerufen")
                return data
                
            except Exception as e:
                logger.error(f"Fehler beim Abruf von {symbol} (Versuch {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        return None

    def _check_api_rate_limit(self) -> bool:
        """Prüft, ob das API-Limit erreicht wurde"""
        rate_limit = int(os.getenv("API_RATE_LIMIT", 30))
        current_time = time.time()
        
        # Entferne alte Anfragen aus der Historie
        self.api_request_history = [
            timestamp for timestamp in self.api_request_history
            if current_time - timestamp < 60  # Letzte Minute
        ]
        
        return len(self.api_request_history) < rate_limit

    def _get_rate_limit_wait_time(self) -> int:
        """Berechnet die Wartezeit bis zum nächsten API-Aufruf"""
        if not self.api_request_history:
            return 0
        
        oldest_request = min(self.api_request_history)
        current_time = time.time()
        
        # Warte bis die älteste Anfrage 60 Sekunden alt ist
        wait_time = max(0, 60 - (current_time - oldest_request))
        return wait_time 