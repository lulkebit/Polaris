from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from utils.logger import logger

# Initialisiere Sentiment-Analyse
logger.info("Initialisiere Sentiment-Analyzer")
sia = SentimentIntensityAnalyzer()

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starte Sentiment-Analyse für {len(df)} Nachrichtentitel")
    
    try:
        df["sentiment"] = df["title"].apply(lambda x: sia.polarity_scores(x)["compound"])
        
        # Logge Verteilung der Sentiments
        positive = len(df[df["sentiment"] > 0])
        negative = len(df[df["sentiment"] < 0])
        neutral = len(df[df["sentiment"] == 0])
        logger.info(f"Sentiment-Verteilung: {positive} positiv, {negative} negativ, {neutral} neutral")
        
        return df
    except Exception as e:
        logger.error(f"Fehler bei der Sentiment-Analyse: {str(e)}")
        raise