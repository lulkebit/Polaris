import pandas as pd
from utils.logger import logger

def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starte Bereinigung von {len(df)} Marktdatensätzen")
    
    # Handle missing values
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        logger.warning(f"{original_len - len(df)} Zeilen mit fehlenden Werten entfernt")
    
    # Normalisiere Preise
    logger.info("Führe Preisnormalisierung durch")
    df["close_normalized"] = (df["close"] - df["close"].mean()) / df["close"].std()
    
    logger.info(f"Datenbereinigung abgeschlossen, {len(df)} Datensätze verarbeitet")
    return df