@echo off
echo Trading AI Analysis System
echo ==========================

:: Frage ob Installation übersprungen werden soll
echo.
echo Möchten Sie die Installation überspringen? (j/n)
set /p SKIP_INSTALL="Ihre Wahl (j/n): "

if "%SKIP_INSTALL%"=="j" (
    goto start_analysis
)

:: Prüfe .env Datei
if not exist .env (
    echo Fehler: .env Datei nicht gefunden!
    echo Bitte erstellen Sie eine .env Datei mit den Konfigurationsdaten.
    pause
    exit /b 1
)

:: Prüfe Python-Installation
where python > nul 2>&1
if errorlevel 1 (
    echo [91mFehler: Python wurde nicht gefunden![0m
    echo Bitte installieren Sie Python und fügen Sie es zum PATH hinzu.
    pause
    exit /b 1
)

:: Setup virtuelle Umgebung
call setup_venv.bat
if errorlevel 1 (
    exit /b 1
)

:: Installiere Requirements
call install_reqs.bat
if errorlevel 1 (
    exit /b 1
)

:start_analysis
call start_analysis.bat 