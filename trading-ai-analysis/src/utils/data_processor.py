import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def reduce_dataframe(df: pd.DataFrame, reduction_factor: float) -> pd.DataFrame:
    """
    Reduziert die Größe des DataFrames durch Auswahl jeder n-ten Zeile.
    
    Args:
        df: DataFrame das reduziert werden soll
        reduction_factor: Faktor um den reduziert werden soll (0.1 = 10% der Daten behalten)
        
    Returns:
        pd.DataFrame: Reduziertes DataFrame
    """
    if reduction_factor >= 1.0:
        return df
        
    step = int(1 / reduction_factor)
    return df.iloc[::step].copy()

def calculate_technical_indicators(df, reduction_factor: float = 1.0):
    """
    Berechnet technische Indikatoren mit optionaler Datenreduktion.
    
    Args:
        df: DataFrame mit den Rohdaten
        reduction_factor: Faktor für die Datenreduktion (default: 1.0 = keine Reduktion)
    """
    # Daten reduzieren wenn nötig
    df = reduce_dataframe(df, reduction_factor)
    
    # Moving Averages
    ma20 = SMAIndicator(close=df['close'], window=20)
    ma50 = SMAIndicator(close=df['close'], window=50)
    df['ma_20'] = ma20.sma_indicator()
    df['ma_50'] = ma50.sma_indicator()
    
    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    
    return df.dropna() 