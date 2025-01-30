import requests
import pandas as pd
from dotenv import load_dotenv
import os
from storage.database import DatabaseConnection
from utils.logger import logger
from sqlalchemy import text

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(query: str = "stock market") -> pd.DataFrame:
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    
    # Verarbeite Artikel
    df = pd.DataFrame(articles)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    return df[["title", "description", "publishedAt", "url"]]

def save_news_to_db(symbol: str, news_df: pd.DataFrame):
    """Speichert verarbeitete News in der Datenbank"""
    db = DatabaseConnection()
    
    try:
        with db.engine.connect() as connection:
            for _, row in news_df.iterrows():
                connection.execute(
                    text("""
                        INSERT INTO news_data (symbol, title, description, published_at, url, sentiment)
                        VALUES (:symbol, :title, :description, :published_at, :url, :sentiment)
                    """),
                    {
                        "symbol": symbol,
                        "title": row["title"],
                        "description": row["description"],
                        "published_at": row["publishedAt"].isoformat(),
                        "url": row["url"],
                        "sentiment": row["sentiment"]
                    }
                )
            connection.commit()
            logger.info(f"{len(news_df)} News-Einträge für {symbol} gespeichert")
            
    except Exception as e:
        logger.error(f"Fehler beim Speichern der News für {symbol}: {str(e)}")
        raise

# Beispielaufruf
if __name__ == "__main__":
    news_df = fetch_news()
    print(news_df.head())