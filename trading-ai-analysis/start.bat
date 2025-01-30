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

:: Prüfe Python-Installation
where python > nul 2>&1
if errorlevel 1 (
    echo [91mFehler: Python wurde nicht gefunden![0m
    echo Bitte installieren Sie Python und fügen Sie es zum PATH hinzu.
    pause
    exit /b 1
)

:: Lösche alte virtuelle Umgebung wenn vorhanden und erstelle neue
if exist venv (
    echo Virtuelle Umgebung gefunden.
    choice /c jn /m "Möchten Sie die bestehende virtuelle Umgebung löschen?"
    if errorlevel 2 (
        echo Bestehende virtuelle Umgebung wird beibehalten.
        goto :activate_venv
    ) else (
        echo Lösche alte virtuelle Umgebung...
        rmdir /s /q venv
        echo Erstelle neue virtuelle Umgebung...
        python -m venv venv
    )
) else (
    echo Erstelle neue virtuelle Umgebung...
    python -m venv venv
)

:activate_venv
:: Aktiviere virtuelle Umgebung
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [91mFehler beim Aktivieren der virtuellen Umgebung![0m
    pause
    exit /b 1
)

:: Installiere pip manuell
::echo [92m[1/7][0m Installiere pip...
::curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
::python get-pip.py --force-reinstall
::del get-pip.py

:: Aktualisiere pip und installiere Basispakete
echo [92m[2/7][0m Aktualisiere pip...
python -m pip install --upgrade pip

echo [92m[3/7][0m Installiere Build-Tools...
python -m pip install --upgrade setuptools wheel build pip-tools
python -m pip install --upgrade pip setuptools

echo [92m[4/7][0m Installiere grundlegende Dependencies...
pip install requests packaging pyyaml regex huggingface-hub safetensors

echo [92m[5/7][0m Installiere PyTorch...
pip install torch --index-url https://download.pytorch.org/whl/cu118

echo [92m[6/7][0m Installiere kritische Pakete...
pip install sqlalchemy pandas numpy python-dotenv psycopg2-binary transformers

echo [92m[7/7][0m Installiere weitere Requirements...
pip install scikit-learn plotly schedule pytest black mypy python-dateutil requests
:: Installiere ta mit allen Dependencies
pip install numpy pandas
pip install ta
pip install -r requirements.txt

:: Verifiziere Installation
echo.
echo Verifiziere Installation...
python -c "import torch; import sqlalchemy; import pandas; import numpy" 2>nul
if errorlevel 1 (
    echo [91mFehler: Kritische Pakete konnten nicht importiert werden![0m
    echo Bitte prüfen Sie die Fehlermeldungen oben.
    pause
    exit /b 1
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
python -c "import torch; cuda_available = torch.cuda.is_available(); print('CUDA Version:', torch.version.cuda if cuda_available else 'Nicht verfügbar'); print('Verfügbare GPUs:', torch.cuda.device_count() if cuda_available else 0); print('Aktive GPU:', torch.cuda.get_device_name(0) if cuda_available else 'Keine'); import os; os.environ['MODEL_DEVICE'] = 'cuda' if cuda_available else 'cpu'" 2>nul
if errorlevel 1 (
    echo [93mWarnung: GPU-Status konnte nicht geprüft werden[0m
    echo [93mSystem läuft im CPU-Modus[0m
    set MODEL_DEVICE=cpu
) else (
    for /f "tokens=*" %%i in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"') do set MODEL_DEVICE=%%i
)
echo Verwende Gerät: %MODEL_DEVICE%
echo.

:: Prüfe Datenbankverbindung
echo Prüfe Datenbankverbindung...
python -c "from sqlalchemy import create_engine; from dotenv import load_dotenv; import os; load_dotenv(); params = {'host': os.getenv('DB_HOST', 'localhost'), 'port': os.getenv('DB_PORT', '5432'), 'database': os.getenv('DB_NAME'), 'user': os.getenv('DB_USER'), 'password': os.getenv('DB_PASSWORD')}; engine = create_engine(f'postgresql://{params[\"user\"]}:{params[\"password\"]}@{params[\"host\"]}:{params[\"port\"]}/{params[\"database\"]}'); engine.connect()" 2>nul
if errorlevel 1 (
    echo [91mFehler: Keine Verbindung zur Datenbank möglich![0m
    echo Bitte überprüfen Sie die Datenbankeinstellungen in der .env Datei.
    pause
    exit /b 1
)

echo [92mDatenbankverbindung erfolgreich hergestellt[0m
echo.

:: Frage nach Analysemodus
echo.
echo Wählen Sie den Analysemodus:
echo [1] Neue Analyse durchführen
echo [2] Vorhandene Daten aus der Datenbank verwenden
set /p ANALYSIS_MODE="Ihre Wahl (1-2): "

if "%ANALYSIS_MODE%"=="1" (
    set ANALYSIS_MODE=new
    echo Neue Analyse wird durchgeführt...
) else if "%ANALYSIS_MODE%"=="2" (
    set ANALYSIS_MODE=last
    echo Vorhandene Daten werden verwendet...
) else (
    set ANALYSIS_MODE=new
    echo Ungültige Eingabe - Neue Analyse wird durchgeführt...
)
echo.

:: Frage nach Performance-Modus
echo Wählen Sie den Performance-Modus:
echo [1] Normal     (Beste Qualität, hoher Ressourcenverbrauch)
echo [2] Low        (Gute Qualität, reduzierter Ressourcenverbrauch)
echo [3] Ultra-Low  (Basis-Qualität, minimaler Ressourcenverbrauch)
echo [4] Auto       (Automatische Anpassung basierend auf System)
set /p PERFORMANCE_MODE="Ihre Wahl (1-4): "

if "%PERFORMANCE_MODE%"=="1" (
    set PERFORMANCE_MODE=normal
    echo Normal-Performance-Modus aktiviert
) else if "%PERFORMANCE_MODE%"=="2" (
    set PERFORMANCE_MODE=low
    echo Low-Performance-Modus aktiviert
) else if "%PERFORMANCE_MODE%"=="3" (
    set PERFORMANCE_MODE=ultra-low
    echo Ultra-Low-Performance-Modus aktiviert
) else if "%PERFORMANCE_MODE%"=="4" (
    set PERFORMANCE_MODE=auto
    echo Automatischer Performance-Modus aktiviert
) else (
    set PERFORMANCE_MODE=normal
    echo Ungültige Eingabe - Normal-Performance-Modus aktiviert
)
echo.

:: Starte KI-Analyse
echo Starte KI-Analyse-System...
echo - Deepseek 1.3B Modell wird geladen
echo - Verbindung zur Datenbank hergestellt
echo - Analyse-Scheduler wird gestartet
echo.

set PYTHONPATH=%CD%\src
python src/main.py --mode %ANALYSIS_MODE% --performance %PERFORMANCE_MODE%

echo.
echo Trading AI Analysis läuft...
echo Drücken Sie STRG+C zum Beenden.
echo.

cmd /k 