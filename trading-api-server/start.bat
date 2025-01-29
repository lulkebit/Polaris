@echo off
setlocal enabledelayedexpansion

:: Setze Titel
title Trading API Server

:: Farbkonfiguration
set "BLUE=[94m"
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "RESET=[0m"

:: Banner anzeigen
echo %BLUE%
echo  _____              _ _                  _    ____ ___   ____                           
echo ^|_   _^|_ _ __ _  __^| (_)_ __   __ _    / \  ^|  _ \_ _^| / ___^|___ _ ____   _____ _ __ 
echo   ^| ^|/ _` / _` ^|/ _` ^| ^| '_ \ / _` ^|  / _ \ ^| ^|_) ^| ^| ^| ^|   / _ \ '__\ \ / / _ \ '__^|
echo   ^| ^| (_^| ^| (_^| ^| (_^| ^| ^| ^| ^| ^| (_^| ^| / ___ \^|  __/^| ^| ^| ^|__^| (_) ^| ^|  \ V /  __/ ^|   
echo   ^|_^|\__,_\__,_^|\__,_^|_^|_^| ^|_^|\__, ^|/_/   \_\_^|  ^|___^| \____\___/^|_^|   \_/ \___^|_^|   
echo                                ^|___/                                                    
echo %RESET%

:: Überprüfe Python-Installation
python --version > nul 2>&1
if errorlevel 1 (
    echo %RED%Python ist nicht installiert oder nicht im PATH!%RESET%
    echo Bitte installiere Python von https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Überprüfe, ob venv existiert
if not exist "venv" (
    echo %YELLOW%Erstelle neue virtuelle Umgebung...%RESET%
    python -m venv venv
    if errorlevel 1 (
        echo %RED%Fehler beim Erstellen der virtuellen Umgebung!%RESET%
        pause
        exit /b 1
    )
)

:: Aktiviere venv
call venv\Scripts\activate
if errorlevel 1 (
    echo %RED%Fehler beim Aktivieren der virtuellen Umgebung!%RESET%
    pause
    exit /b 1
)

:: Installiere/Aktualisiere Abhängigkeiten
echo %YELLOW%Installiere/Aktualisiere Abhängigkeiten...%RESET%
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo %RED%Fehler beim Installieren der Abhängigkeiten!%RESET%
    pause
    exit /b 1
)

:: Erstelle logs Verzeichnis
if not exist "logs" mkdir logs

:: Starte den Server
echo %GREEN%Starte den API-Server...%RESET%
python src/main.py

:: Bei Fehler
if errorlevel 1 (
    echo %RED%Fehler beim Starten des Servers!%RESET%
    pause
)

:: Deaktiviere venv
deactivate 