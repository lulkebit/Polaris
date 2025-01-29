# Polaris Trading AI Analysis

## Übersicht

Die KI-Analysekomponente der Polaris Trading Platform verwendet fortschrittliche Modelle für die Analyse von Marktdaten und Finanznachrichten. Das Herzstück ist das Deepseek 1.3B Modell, das für die Verarbeitung von Finanzdaten optimiert wurde.

## Technologie-Stack

-   Python 3.10+
-   PyTorch
-   Deepseek 1.3B
-   Pandas
-   NumPy
-   FastAPI (für API-Endpunkte)
-   SQLAlchemy

## Projektstruktur

```
trading-ai-analysis/
├── src/
│   ├── models/           # KI-Modelle und Inferenz
│   │   ├── deepseek/    # Deepseek-spezifische Implementierung
│   │   └── utils/       # Modell-Hilfsfunktionen
│   ├── analysis/        # Analysekomponenten
│   │   ├── market/      # Marktdatenanalyse
│   │   ├── sentiment/   # Sentiment-Analyse
│   │   └── signals/     # Signalgenerierung
│   ├── data/           # Datenzugriff und -verarbeitung
│   └── utils/          # Allgemeine Hilfsfunktionen
├── tests/             # Testfälle
├── config/            # Konfigurationsdateien
└── requirements.txt   # Projektabhängigkeiten
```

## Installation

1. Python 3.10 oder höher installieren
2. CUDA-fähige GPU einrichten (empfohlen)
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
    MODEL_PATH=/path/to/deepseek/model
    CUDA_VISIBLE_DEVICES=0
    API_PORT=8001
    ```

## Verwendung

### Marktanalyse

```python
from src.analysis.market.analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
analysis = analyzer.analyze_market_data(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback_periods=24
)
```

### Sentiment-Analyse

```python
from src.analysis.sentiment.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze_news(
    news_data="...",
    context="crypto"
)
```

### Signal-Generierung

```python
from src.analysis.signals.generator import SignalGenerator

generator = SignalGenerator()
signals = generator.generate_signals(
    market_data="...",
    sentiment_data="..."
)
```

## API-Endpunkte

Die KI-Komponente stellt RESTful API-Endpunkte bereit:

-   POST `/api/v1/analyze/market`
-   POST `/api/v1/analyze/sentiment`
-   POST `/api/v1/generate/signals`

## Performance-Optimierung

-   GPU-Beschleunigung für Modell-Inferenz
-   Batch-Verarbeitung für erhöhten Durchsatz
-   Caching-Strategien für häufig verwendete Daten

## Logging

Die Logs werden im `logs/` Verzeichnis gespeichert:

-   `model.log` - Modell-Inferenz und Performance
-   `analysis.log` - Analysedetails
-   `api.log` - API-Zugriffe und Fehler

## Tests

Tests ausführen:

```bash
pytest tests/
```

## Lizenz

Proprietär - Alle Rechte vorbehalten
