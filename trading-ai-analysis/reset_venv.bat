@echo off
echo Lösche vorhandene virtuelle Umgebung...
if exist venv (
    rmdir /s /q venv
)

echo Erstelle neue virtuelle Umgebung...
python -m venv venv

echo Aktiviere virtuelle Umgebung...
call venv\Scripts\activate.bat

echo Aktualisiere pip...
python -m pip install --upgrade pip

echo Installiere Abhängigkeiten...
if exist requirements.txt (
    pip install -r requirements.txt
    echo Abhängigkeiten wurden installiert.
) else (
    echo Keine requirements.txt gefunden.
)

echo.
echo Virtuelle Umgebung wurde erfolgreich zurückgesetzt!
echo.

pause 