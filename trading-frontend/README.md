# Polaris Trading Frontend

## Übersicht

Dies ist die Frontend-Komponente der Polaris Trading Platform. Sie bietet eine moderne und intuitive Benutzeroberfläche für die Überwachung und Steuerung des automatisierten Handelssystems.

## Technologie-Stack

-   React 18
-   TypeScript
-   Vite
-   TailwindCSS
-   React Query
-   React Router
-   Recharts für Visualisierungen

## Projektstruktur

```
trading-frontend/
├── src/
│   ├── components/     # Wiederverwendbare UI-Komponenten
│   ├── pages/         # Seitenkomponenten
│   ├── services/      # API-Integration und Dienste
│   ├── hooks/         # Custom React Hooks
│   ├── utils/         # Hilfsfunktionen
│   ├── types/         # TypeScript Definitionen
│   └── assets/        # Statische Ressourcen
├── public/           # Öffentliche Dateien
└── package.json     # Projektabhängigkeiten
```

## Installation

1. Node.js (Version 18 oder höher) installieren
2. Abhängigkeiten installieren:
    ```bash
    npm install
    ```
3. `.env` Datei konfigurieren:
    ```
    VITE_API_URL=http://localhost:8000
    ```

## Entwicklung

Entwicklungsserver starten:

```bash
npm run dev
```

## Build

Produktions-Build erstellen:

```bash
npm run build
```

## Features

-   📊 Echtzeit-Marktdaten-Visualisierung
-   💼 Portfolio-Übersicht und -Management
-   📈 Trading-Strategien-Konfiguration
-   🔔 Benachrichtigungssystem
-   📱 Responsive Design

## API-Integration

Die Frontend-Anwendung kommuniziert mit dem Trading-API-Server über RESTful Endpunkte. Die Basis-URL wird in der `.env` Datei konfiguriert.

## Sicherheit

-   🔒 JWT-basierte Authentifizierung
-   🛡️ CORS-Konfiguration
-   🔐 Sichere API-Key-Verwaltung

## Lizenz

Proprietär - Alle Rechte vorbehalten
