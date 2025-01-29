# Polaris Trading Frontend

Dieses Frontend-Projekt ist Teil des Polaris Trading Systems und bietet eine moderne, benutzerfreundliche Oberfläche für die Visualisierung und Analyse von Handelsdaten.

## Features

-   Echtzeit-Darstellung von Marktdaten
-   Interaktive Trading-Charts
-   Analyse-Dashboard
-   News-Integration
-   Benutzerfreundliche Konfiguration von Trading-Strategien

## Technologie-Stack

-   React.js
-   TypeScript
-   Material-UI
-   Chart.js/TradingView
-   WebSocket für Echtzeit-Updates

## Installation

1. Repository klonen:

```bash
git clone [repository-url]
cd frontend
```

2. Dependencies installieren:

```bash
npm install
```

3. Entwicklungsserver starten:

```bash
npm start
```

## Entwicklung

-   Der Entwicklungsserver läuft standardmäßig auf `http://localhost:3000`
-   Hot-Reloading ist aktiviert
-   API-Endpunkte können in der `.env` Datei konfiguriert werden

## Build

Für einen Produktions-Build:

```bash
npm run build
```

## Projektstruktur

```
frontend/
├── src/
│   ├── components/     # React Komponenten
│   ├── pages/         # Seiten/Routen
│   ├── services/      # API Services
│   ├── store/         # State Management
│   ├── utils/         # Hilfsfunktionen
│   └── App.tsx        # Root Component
├── public/            # Statische Assets
└── package.json       # Projekt-Konfiguration
```

## Verbindung mit Backend

Das Frontend kommuniziert mit dem Trading-Data-Pipeline und Trading-AI-Analysis Backend über RESTful APIs und WebSocket-Verbindungen für Echtzeit-Updates.

## Beitragen

1. Fork des Repositories erstellen
2. Feature Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add some AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request erstellen

## Lizenz

[MIT](https://choosealicense.com/licenses/mit/)
