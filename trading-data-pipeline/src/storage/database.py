from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from utils.logger import logger
import os
import pandas as pd

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # z.B. "postgresql://user:password@localhost:5432/trading_db"
logger.info(f"Initialisiere Datenbankverbindung zu {DATABASE_URL.split('@')[-1]}")  # Logge nur Host/DB, nicht Credentials

try:
    engine = create_engine(DATABASE_URL)
    # Teste die Verbindung
    with engine.connect() as connection:
        logger.info("Datenbankverbindung erfolgreich hergestellt")
except Exception as e:
    logger.error(f"Fehler beim Verbinden zur Datenbank: {str(e)}")
    raise

Session = sessionmaker(bind=engine)

def save_to_database(df: pd.DataFrame, table_name: str, chunk_size: int = 1000):
    """
    Speichert einen DataFrame in der Datenbank mit Batch-Processing.
    
    Args:
        df: Der zu speichernde DataFrame
        table_name: Name der Zieltabelle
        chunk_size: Anzahl der Zeilen pro Batch (Standard: 1000)
    """
    try:
        total_rows = len(df)
        logger.info(f"Starte Speichervorgang für Tabelle {table_name} mit {total_rows} Datensätzen")
        logger.info(f"Datenframe Spalten: {df.columns.tolist()}")
        
        # Prüfe ob die Tabelle bereits existiert
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"))
            exists = result.scalar()
            
            if exists:
                logger.info(f"Lösche existierende Tabelle {table_name}")
                connection.execute(text(f"DROP TABLE {table_name}"))
                connection.commit()
                logger.info(f"Tabelle {table_name} erfolgreich gelöscht")
            
            logger.info(f"Erstelle neue Tabelle {table_name}")

        # Speichere die Daten in Batches
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        logger.info(f"Starte Batch-Processing mit {total_chunks} Chunks (Chunk-Größe: {chunk_size})")
        
        for i, chunk_start in enumerate(range(0, total_rows, chunk_size)):
            chunk = df.iloc[chunk_start:chunk_start + chunk_size]
            if_exists = "replace" if i == 0 else "append"
            
            chunk.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=True,
                index_label="timestamp",
                method='multi'  # Schnellere Methode für große Datensätze
            )
            
            logger.info(f"Chunk {i+1}/{total_chunks} verarbeitet ({len(chunk)} Zeilen)")
        
        # Prüfe ob alle Daten gespeichert wurden
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            logger.info(f"Gesamtanzahl der gespeicherten Datensätze in {table_name}: {count}")
            
            if count == total_rows:
                logger.info(f"[OK] Alle {total_rows} Datensätze erfolgreich gespeichert")
            else:
                logger.warning(f"[!] Anzahl der gespeicherten Datensätze ({count}) weicht von der Eingabe ({total_rows}) ab")
            
            # Zeige Beispieldaten
            sample = connection.execute(text(f"SELECT * FROM {table_name} LIMIT 3")).fetchall()
            if sample:
                logger.info(f"Beispieldaten aus {table_name}: {sample[0]}")
        
    except Exception as e:
        logger.error(f"Fehler beim Speichern in Tabelle {table_name}: {str(e)}")
        raise