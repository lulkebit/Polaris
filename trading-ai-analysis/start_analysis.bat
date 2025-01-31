@echo off
echo Starte Trading AI Analysis...
echo ===========================

:: Aktiviere virtuelle Umgebung
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [91mFehler beim Aktivieren der virtuellen Umgebung![0m
    pause
    exit /b 1
)

:: Frage nach Performance-Modus
echo.
echo Wählen Sie den Performance-Modus:
echo [1] Ultra-Low (langsamste, geringste Ressourcen)
echo [2] Low (langsam, geringe Ressourcen)
echo [3] Normal (empfohlen)
echo [4] High (schnell, hohe Ressourcen)
set /p PERF_MODE="Ihre Wahl (1-4): "

if "%PERF_MODE%"=="1" (
    set PERF_MODE=ultra-low
) else if "%PERF_MODE%"=="2" (
    set PERF_MODE=low
) else if "%PERF_MODE%"=="3" (
    set PERF_MODE=normal
) else if "%PERF_MODE%"=="4" (
    set PERF_MODE=high
) else (
    set PERF_MODE=normal
    echo Ungültige Eingabe - Verwende Normal-Modus...
)

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

:: Starte KI-Analyse
set PYTHONPATH=%CD%\src
python src/main.py --mode %ANALYSIS_MODE% --performance %PERF_MODE%

echo.
echo Trading AI Analysis läuft...
echo Drücken Sie STRG+C zum Beenden.
echo.

cmd /k 