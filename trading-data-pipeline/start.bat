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
    echo Erstelle neue virtuelle Umgebung...
    python -m venv venv
    call venv\Scripts\activate.bat
)

:: Prüfe und installiere Requirements
echo Prüfe Python-Pakete...
python -c "import pkg_resources, sys; pkg_list = [dist.project_name for dist in pkg_resources.working_set]; required = [line.strip().split('==')[0] for line in open('requirements.txt')]; missing = [pkg for pkg in required if pkg.lower() not in [pkg.lower() for pkg in pkg_list]]; sys.exit(1 if missing else 0)"

if errorlevel 1 (
    echo Installiere fehlende Pakete...
    echo [92m[1/3][0m Aktualisiere pip...
    python -m pip install --upgrade pip
    echo [92m[2/3][0m Installiere Requirements...
    pip install -r requirements.txt
    echo [92m[3/3][0m Installation abgeschlossen
) else (
    echo [92mAlle Pakete sind bereits installiert[0m
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