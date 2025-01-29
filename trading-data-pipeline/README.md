# Polaris Trading Data Pipeline

## Übersicht

Die Datenpipeline-Komponente der Polaris Trading Platform ist verantwortlich für die Sammlung, Verarbeitung und Speicherung von Marktdaten und Finanznachrichten. Sie bildet das Fundament für die nachgelagerte KI-Analyse und Handelsentscheidungen.

## Technologie-Stack

-   Python 3.10+
-   Pandas
-   NumPy
-   PostgreSQL/TimescaleDB
-   Apache Kafka (für Event-Streaming)
-   FastAPI (für API-Endpunkte)
-   SQLAlchemy
-   ccxt (für Börsen-Integration)

## Projektstruktur

```
trading-data-pipeline/
├── src/
│   ├── data_collection/      # Datensammlung
│   │   ├── market/          # Marktdaten-Collector
│   │   ├── news/           # Nachrichten-Collector
│   │   └── websockets/     # WebSocket-Verbindungen
│   ├── data_processing/     # Datenverarbeitung
│   │   ├── cleaning/       # Datenbereinigung
│   │   ├── aggregation/    # Datenaggregation
│   │   └── normalization/  # Datennormalisierung
│   ├── storage/            # Datenspeicherung
│   │   ├── database/       # Datenbankoperationen
│   │   ├── streaming/      # Kafka-Integration
│   │   └── cache/         # Caching-Mechanismen
│   └── utils/             # Hilfsfunktionen
├── tests/                # Testfälle
├── config/              # Konfigurationsdateien
└── requirements.txt    # Projektabhängigkeiten
```

## Installation

1. Python 3.10 oder höher installieren
2. PostgreSQL und TimescaleDB einrichten
3. Apache Kafka installieren (optional für Event-Streaming)
4. Virtuelle Umgebung erstellen und aktivieren:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate   # Windows
    ```
5. Abhängigkeiten installieren:
    ```bash
    pip install -r requirements.txt
    ```
6. `.env` Datei konfigurieren:

    ```
    # Datenbank
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=polaris_data
    DB_USER=your_user
    DB_PASSWORD=your_password

    # Kafka (optional)
    KAFKA_BOOTSTRAP_SERVERS=localhost:9092

    # APIs
    BINANCE_API_KEY=your_api_key
    BINANCE_API_SECRET=your_api_secret

    # Server
    API_PORT=8002
    ```

## Verwendung

### Marktdaten-Collection

```python
from src.data_collection.market.collector import MarketDataCollector

collector = MarketDataCollector()
data = collector.collect_market_data(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1m",
    limit=1000
)
```

### Nachrichten-Collection

```python
from src.data_collection.news.collector import NewsCollector

collector = NewsCollector()
news = collector.collect_news(
    sources=["cryptopanic", "twitter"],
    keywords=["bitcoin", "crypto"],
    limit=100
)
```

### Datenverarbeitung

```python
from src.data_processing.processor import DataProcessor

processor = DataProcessor()
processed_data = processor.process_market_data(
    raw_data=data,
    clean=True,
    normalize=True
)
```

## API-Endpunkte

Die Pipeline stellt RESTful API-Endpunkte bereit:

-   GET `/api/v1/market/data`
-   GET `/api/v1/market/latest`
-   GET `/api/v1/news/feed`
-   POST `/api/v1/data/process`

## Performance-Optimierung

-   Effiziente Datenbankindizes
-   Materialisierte Views für häufige Abfragen
-   Streaming-Verarbeitung für Echtzeitdaten
-   Caching-Strategien für häufig verwendete Daten

## Datenbank-Schema

Die wichtigsten Tabellen:

-   `market_data`: Marktdaten (OHLCV)
-   `trades`: Ausgeführte Trades
-   `news_feed`: Finanznachrichten
-   `technical_indicators`: Technische Indikatoren

## Logging

Die Logs werden im `logs/` Verzeichnis gespeichert:

-   `collector.log` - Datensammlung
-   `processor.log` - Datenverarbeitung
-   `storage.log` - Datenbankoperationen
-   `api.log` - API-Zugriffe

## Tests

Tests ausführen:

```bash
pytest tests/
```

## Monitoring

Die Pipeline verfügt über verschiedene Monitoring-Metriken:

-   Datensammlung-Performance
-   Verarbeitungszeiten
-   Datenbankauslastung
-   API-Latenz

## Fehlerbehandlung

-   Automatische Wiederverbindung bei API-Ausfällen
-   Daten-Validierung und -Bereinigung
-   Backup-Datenquellen
-   Fehlerbenachrichtigungen

## Lizenz

Proprietär - Alle Rechte vorbehalten
