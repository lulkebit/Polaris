# Polaris Trading API Server

## Übersicht

Der Trading-API-Server ist die zentrale Ausführungskomponente der Polaris Trading Platform. Er verarbeitet Handelssignale, führt Trades aus und koordiniert die Kommunikation zwischen der KI-Analyse, der Datenpipeline und dem Frontend.

## Technologie-Stack

-   Python 3.10+
-   FastAPI
-   SQLAlchemy
-   Redis (für Caching)
-   JWT (für Authentifizierung)
-   ccxt (für Börsen-Integration)
-   WebSocket (für Echtzeit-Updates)
-   Pydantic (für Datenvalidierung)

## Projektstruktur

```
trading-api-server/
├── src/
│   ├── api/                # API-Endpunkte
│   │   ├── v1/            # API Version 1
│   │   ├── ws/            # WebSocket-Handler
│   │   └── auth/          # Authentifizierung
│   ├── services/          # Geschäftslogik
│   │   ├── trading/       # Trading-Engine
│   │   ├── risk/          # Risikomanagement
│   │   ├── portfolio/     # Portfolio-Management
│   │   └── signals/       # Signal-Verarbeitung
│   ├── models/           # Datenmodelle
│   │   ├── db/           # Datenbankmodelle
│   │   └── schema/       # API-Schemas
│   └── utils/            # Hilfsfunktionen
├── tests/               # Testfälle
├── config/             # Konfigurationsdateien
└── requirements.txt   # Projektabhängigkeiten
```

## Installation

1. Python 3.10 oder höher installieren
2. Redis installieren und starten
3. Virtuelle Umgebung erstellen und aktivieren:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate   # Windows
    ```
4. Abhängigkeiten installieren:
    ```bash
    pip install -r requirements.txt
    ```
5. `.env` Datei konfigurieren:

    ```
    # Server
    API_HOST=localhost
    API_PORT=8000
    DEBUG=True

    # Datenbank
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=polaris_trading
    DB_USER=your_user
    DB_PASSWORD=your_password

    # Redis
    REDIS_HOST=localhost
    REDIS_PORT=6379

    # JWT
    JWT_SECRET=your_secret_key
    JWT_ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=30

    # Börsen
    BINANCE_API_KEY=your_api_key
    BINANCE_API_SECRET=your_api_secret

    # Externe Services
    AI_SERVICE_URL=http://localhost:8001
    DATA_SERVICE_URL=http://localhost:8002
    ```

## API-Endpunkte

### Trading

-   POST `/api/v1/trades/execute` - Trade ausführen
-   GET `/api/v1/trades/active` - Aktive Trades abrufen
-   GET `/api/v1/trades/history` - Trade-Historie abrufen
-   DELETE `/api/v1/trades/{trade_id}` - Trade abbrechen

### Portfolio

-   GET `/api/v1/portfolio/overview` - Portfolio-Übersicht
-   GET `/api/v1/portfolio/positions` - Offene Positionen
-   GET `/api/v1/portfolio/performance` - Performance-Metriken

### Signale

-   POST `/api/v1/signals/process` - Handelssignal verarbeiten
-   GET `/api/v1/signals/active` - Aktive Signale abrufen

### WebSocket

-   `/ws/v1/market` - Echtzeit-Marktdaten
-   `/ws/v1/trades` - Echtzeit-Trade-Updates
-   `/ws/v1/signals` - Echtzeit-Handelssignale

## Trading-Engine

Die Trading-Engine unterstützt:

-   Market Orders
-   Limit Orders
-   Stop-Loss Orders
-   Take-Profit Orders
-   OCO (One-Cancels-Other) Orders

## Risikomanagement

Integrierte Risikomanagement-Funktionen:

-   Position Sizing
-   Stop-Loss-Management
-   Exposure-Limits
-   Drawdown-Protection
-   Volatilitäts-basierte Anpassungen

## Performance-Optimierung

-   Redis-Caching für häufige Abfragen
-   Asynchrone Verarbeitung
-   Connection Pooling
-   Rate Limiting
-   Request Queuing

## Sicherheit

-   JWT-basierte Authentifizierung
-   API-Key-Verschlüsselung
-   Rate Limiting
-   IP-Whitelist
-   CORS-Konfiguration
-   Request-Validierung

## Logging

Die Logs werden im `logs/` Verzeichnis gespeichert:

-   `api.log` - API-Zugriffe
-   `trading.log` - Trading-Aktivitäten
-   `error.log` - Fehler und Warnungen
-   `performance.log` - Performance-Metriken

## Tests

Tests ausführen:

```bash
pytest tests/
```

## Monitoring

Echtzeit-Monitoring für:

-   API-Performance
-   Trading-Aktivitäten
-   Systemressourcen
-   Fehlerraten
-   Latenzzeiten

## Fehlerbehandlung

-   Automatische Wiederverbindung
-   Circuit Breaker Pattern
-   Fallback-Strategien
-   Error Reporting
-   Automatische Benachrichtigungen

## Entwicklung

Entwicklungsserver starten:

```bash
uvicorn src.main:app --reload --port 8000
```

API-Dokumentation aufrufen:

```
http://localhost:8000/docs
```

## Lizenz

Proprietär - Alle Rechte vorbehalten
