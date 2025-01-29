# Trading AI Analysis

Dieses Projekt enthält die KI-Komponenten für die Analyse von Handelsdaten und Finanznachrichten.

## Projektstruktur

```
trading-ai-analysis/
├── src/
│   ├── models/          # KI-Modelle
│   ├── utils/           # Hilfsfunktionen
│   └── analysis/        # Analysekomponenten
├── tests/              # Testfälle
├── config/             # Konfigurationsdateien
└── requirements.txt    # Projektabhängigkeiten
```

## Installation

1. Erstellen Sie eine virtuelle Umgebung:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Installieren Sie die Abhängigkeiten:

```bash
pip install -r requirements.txt
```

## Verwendung

Das Projekt bietet KI-gestützte Analysen für:

-   Marktdaten
-   Finanznachrichten
-   Kombinierte Analysen

Beispiel:

```python
from src.analysis.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
analysis = analyzer.analyze_data(market_data="...", news_data="...")
print(analysis)
```

## Logging

Die Logs werden im `logs/` Verzeichnis gespeichert und enthalten detaillierte Informationen über die Ausführung der Analysen.
