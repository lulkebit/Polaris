import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def calculate_technical_indicators(df):
    """Berechnet technische Indikatoren"""
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