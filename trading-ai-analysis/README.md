# Trading AI Analysis System

Ein KI-gestütztes System zur Analyse, Optimierung und Backtesting von Handelsstrategien.

## Features

-   **Marktanalyse**

    -   Technische Indikatoren
    -   Sentiment-Analyse
    -   Trendanalyse
    -   KI-gestützte Markteinschätzung

-   **Risikomanagement**

    -   Position Sizing
    -   Drawdown-Kontrolle
    -   Sektor-Exposure
    -   VaR-Berechnung
    -   Stress-Tests

-   **Backtesting**

    -   Historische Performance-Analyse
    -   Transaktionskosten
    -   Slippage-Simulation
    -   Benchmark-Vergleich

-   **Strategieoptimierung**

    -   KI-gestützte Parameteroptimierung
    -   Genetische Algorithmen
    -   Walk-Forward-Tests
    -   Performance-Metriken

-   **Visualisierung**
    -   Performance-Dashboards
    -   Risiko-Metriken
    -   Trade-Analysen
    -   Portfolio-Zusammensetzung

## Installation

1. Klonen Sie das Repository:

```bash
git clone https://github.com/yourusername/trading-ai-analysis.git
cd trading-ai-analysis
```

2. Erstellen Sie eine `.env` Datei basierend auf `.env.example`:

```bash
cp .env.example .env
```

3. Konfigurieren Sie die Umgebungsvariablen in der `.env` Datei:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=your_user
DB_PASSWORD=your_password
...
```

4. Starten Sie das System:

```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

## Systemanforderungen

-   Python 3.9+
-   PostgreSQL 13+
-   CUDA-fähige GPU (optional, für bessere Performance)
-   Mindestens 16GB RAM
-   100GB freier Festplattenspeicher

## Abhängigkeiten

Hauptabhängigkeiten:

-   numpy>=1.24.0
-   pandas>=2.0.0
-   scikit-learn>=1.3.0
-   plotly>=5.18.0
-   python-dotenv>=1.0.0
-   sqlalchemy>=2.0.0
-   torch>=2.1.0
-   transformers>=4.35.0

Weitere Details finden Sie in `requirements.txt`.

## Projektstruktur

```
trading-ai-analysis/
├── src/
│   ├── analysis/          # Marktanalyse
│   ├── backtesting/       # Backtesting-Engine
│   ├── data_processing/   # Datenverarbeitung
│   ├── models/            # KI-Modelle
│   ├── optimization/      # Strategieoptimierung
│   ├── risk/             # Risikomanagement
│   ├── database/         # Datenbankzugriff
│   └── utils/            # Hilfsfunktionen
├── data/                 # Marktdaten
├── results/             # Analyse-Ergebnisse
├── logs/               # Logdateien
├── tests/             # Testfälle
├── requirements.txt   # Python-Abhängigkeiten
├── .env              # Konfiguration
└── README.md         # Dokumentation
```

## Nutzung

1. **Datenaufbereitung**

```python
from data_processing.data_aggregator import create_combined_market_data

# Lade und bereite Marktdaten vor
market_data = create_combined_market_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

2. **Marktanalyse**

```python
from analysis.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
analysis = analyzer.analyze_data(market_data)
```

3. **Backtesting**

```python
from backtesting.backtester import Backtester, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    commission_rate=0.001
)
backtester = Backtester(config)
results = backtester.run_backtest(market_data)
```

4. **Strategieoptimierung**

```python
from optimization.strategy_optimizer import StrategyOptimizer

optimizer = StrategyOptimizer()
best_strategy, best_results = optimizer.optimize_strategy(market_data)
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der `LICENSE` Datei.

## Beitragen

Beiträge sind willkommen! Bitte lesen Sie `CONTRIBUTING.md` für Details zum Prozess für Pull Requests.
