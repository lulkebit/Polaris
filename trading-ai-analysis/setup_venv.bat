@echo off
echo Verwalte virtuelle Umgebung...
echo ==============================

:: Frage ob alte venv gelöscht werden soll
echo.
echo Möchten Sie die bestehende virtuelle Umgebung löschen? (j/n)
set /p DELETE_VENV="Ihre Wahl (j/n): "

if "%DELETE_VENV%"=="j" (
    echo Lösche alte virtuelle Umgebung...
    rmdir /s /q venv
    echo Erstelle neue virtuelle Umgebung...
    python -m venv venv
) else (
    echo Behalte bestehende virtuelle Umgebung bei.
)

:: Aktiviere venv
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [91mFehler beim Aktivieren der virtuellen Umgebung![0m
    pause
    exit /b 1
)

echo Virtuelle Umgebung erfolgreich eingerichtet!
pause 