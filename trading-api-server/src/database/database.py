from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()

# Datenbankverbindung aus Umgebungsvariablen
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trading_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    
    # Teste die Verbindung
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()")).scalar()
        logger.info(f"Erfolgreich mit PostgreSQL verbunden: {result}")
        
        # Überprüfe die ai_analysis Tabelle
        table_exists = connection.execute(
            text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'ai_analysis'
            )
            """)
        ).scalar()
        
        if table_exists:
            # Überprüfe die Spalten
            columns = connection.execute(
                text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'ai_analysis'
                """)
            ).fetchall()
            logger.info(f"Gefundene Spalten in ai_analysis: {columns}")
        else:
            logger.error("Tabelle 'ai_analysis' nicht gefunden!")
            raise Exception("Tabelle 'ai_analysis' existiert nicht in der Datenbank")
            
except Exception as e:
    logger.error(f"Datenbankfehler: {str(e)}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency für FastAPI, die eine Datenbankverbindung bereitstellt
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 