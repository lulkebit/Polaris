import requests
import pandas as pd
from dotenv import load_dotenv
import os

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

# Beispielaufruf
if __name__ == "__main__":
    news_df = fetch_news()
    print(news_df.head())