@echo off
echo Starte Trading AI Analysis...
echo ===========================

:: Prüfe ob .env existiert
if not exist .env (
    echo Fehler: .env Datei nicht gefunden!
    echo Bitte erstellen Sie eine .env Datei mit den Konfigurationsdaten.
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
    echo [92m[1/4][0m Aktualisiere pip...
    python -m pip install --upgrade pip
    echo [92m[2/4][0m Installiere PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo [92m[3/4][0m Installiere weitere Requirements...
    pip install -r requirements.txt
    echo [92m[4/4][0m Installation abgeschlossen
) else (
    echo [92mAlle Pakete sind bereits installiert[0m
)

:: Zeige Systeminformationen
echo.
echo System-Information:
echo ------------------
python --version
echo Virtuelle Umgebung: %VIRTUAL_ENV%

:: Prüfe CUDA-Verfügbarkeit
echo.
echo GPU-Information:
echo ---------------
python -c "import torch; print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'Nicht verfügbar'); print('Verfügbare GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('Aktive GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Keine')"
if errorlevel 1 (
    echo [93mWarnung: GPU-Status konnte nicht geprüft werden[0m
    echo [93mSystem läuft im CPU-Modus[0m
)
echo.

:: Prüfe Datenbankverbindung
echo Prüfe Datenbankverbindung...
python -c "from sqlalchemy import create_engine; from dotenv import load_dotenv; import os; load_dotenv(); params = {'host': os.getenv('DB_HOST', 'localhost'), 'port': os.getenv('DB_PORT', '5432'), 'database': os.getenv('DB_NAME'), 'user': os.getenv('DB_USER'), 'password': os.getenv('DB_PASSWORD')}; engine = create_engine(f'postgresql://{params[\"user\"]}:{params[\"password\"]}@{params[\"host\"]}:{params[\"port\"]}/{params[\"database\"]}'); engine.connect()"
if errorlevel 1 (
    echo [91mFehler: Keine Verbindung zur Datenbank möglich![0m
    echo Bitte überprüfen Sie die Datenbankeinstellungen in der .env Datei.
    pause
    exit /b 1
)

echo [92mDatenbankverbindung erfolgreich hergestellt[0m
echo.

:: Starte KI-Analyse
echo Starte KI-Analyse-System...
echo - Deepseek 1.3B Modell wird geladen
echo - Verbindung zur Datenbank hergestellt
echo - Analyse-Scheduler wird gestartet
echo.

python src/main.py

echo.
echo Trading AI Analysis läuft...
echo Drücken Sie STRG+C zum Beenden.
echo.

cmd /k 