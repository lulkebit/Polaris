@echo off
setlocal enabledelayedexpansion

title AI Trading Analysis Engine

:: Farbdefinitionen
set "CYAN=[96m"
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "RESET=[0m"

:: Prüfe Python-Installation
python --version > nul 2>&1
if errorlevel 1 (
    echo %RED%Python 3.10+ nicht gefunden!%RESET%
    echo Bitte installieren Sie Python von https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Prüfe .env Datei
if not exist ".env" (
    echo %YELLOW%.env Datei nicht gefunden!%RESET%
    echo Erstellen Sie eine .env Datei mit den Datenbankzugangsdaten
    pause
    exit /b 1
)

:: Virtuelle Umgebung
if not exist "venv" (
    echo %YELLOW%Erstelle virtuelle Umgebung...%RESET%
    python -m venv venv
    if errorlevel 1 (
        echo %RED%Fehler beim Erstellen der virtuellen Umgebung!%RESET%
        pause
        exit /b 1
    )
)

call venv\Scripts\activate
if errorlevel 1 (
    echo %RED%Fehler beim Aktivieren der virtuellen Umgebung!%RESET%
    pause
    exit /b 1
)

:: Installiere Abhängigkeiten
echo %YELLOW%Installiere Pakete...%RESET%
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo %RED%Fehler bei der Paketinstallation!%RESET%
    pause
    exit /b 1
)

:: Erstelle Log-Verzeichnis
if not exist "logs" mkdir logs

:: Starte Analyse-Pipeline
echo %GREEN%Starte KI-Analyse...%RESET%
python src/main.py

if errorlevel 1 (
    echo %RED%Fehler in der Analyse-Pipeline!%RESET%
    pause
    exit /b 1
)

deactivate 