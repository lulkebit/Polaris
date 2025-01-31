@echo off
echo Installiere Python-Pakete...
echo ===========================

:: Aktualisiere pip
echo [1/3] Aktualisiere pip...
python -m pip install --upgrade pip

:: Installiere Basispakete
echo [2/3] Installiere Basispakete...
pip install requests packaging pyyaml regex huggingface-hub safetensors
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install sqlalchemy pandas numpy python-dotenv psycopg2-binary transformers

:: Installiere weitere Requirements
echo [3/3] Installiere weitere Requirements...
pip install scikit-learn plotly schedule pytest black mypy python-dateutil requests
pip install numpy pandas
pip install ta
pip install -r requirements.txt

echo Installation abgeschlossen!
pause 