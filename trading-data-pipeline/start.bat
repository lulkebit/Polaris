@echo off
echo Starte Trading Data Pipeline...
echo ============================

:: Prüfe ob .env existiert
if not exist .env (
    echo Fehler: .env Datei nicht gefunden!
    echo Bitte erstellen Sie eine .env Datei mit den Datenbankzugangsdaten.
    pause
    exit /b 1
)

:: Aktiviere virtuelle Umgebung
call venv\Scripts\activate.bat

:: Prüfe ob Aktivierung erfolgreich war
if errorlevel 1 (
    echo Fehler beim Aktivieren der virtuellen Umgebung!
    pause
    exit /b 1
)

:: Zeige Systeminformationen
echo.
echo System-Information:
echo ------------------
python --version
echo Virtuelle Umgebung: %VIRTUAL_ENV%
echo.

:: Starte die Pipeline
echo Starte Datensammlung...
echo - Marktdaten-Collection
echo - News-Aggregation
echo - Datenverarbeitung
echo.

python src/main.py

echo.
echo Trading Data Pipeline läuft...
echo Drücken Sie STRG+C zum Beenden.
echo.

cmd /k 