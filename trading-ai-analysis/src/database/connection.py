import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import logging
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

from .schema import Base

class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self._engine = None
        self._session_factory = None
        self._initialized = True
        
    def initialize(self, connection_string: Optional[str] = None):
        """Initialisiert die Datenbankverbindung"""
        # Lade Umgebungsvariablen
        load_dotenv()
        
        if not connection_string:
            # Verwende die gleichen Parameter wie in start.bat
            params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
            
            connection_string = f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:{params["port"]}/{params["database"]}'
            
        if not connection_string:
            raise ValueError("Keine Datenbankverbindungs-URL gefunden")
            
        try:
            self._engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            
            # Erstelle alle Tabellen
            Base.metadata.create_all(self._engine)
            
            self._session_factory = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self._engine
                )
            )
            
            # Teste die Verbindung
            self.test_connection()
            self.logger.info("Datenbankverbindung erfolgreich initialisiert")
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenbankinitialisierung: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self):
        """Stellt eine Datenbanksession im Context Manager bereit"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self):
        """Schließt alle Datenbankverbindungen"""
        if self._session_factory:
            self._session_factory.remove()
        if self._engine:
            self._engine.dispose()
            
    @property
    def engine(self):
        """Gibt die Engine-Instanz zurück"""
        if not self._engine:
            raise RuntimeError("Datenbankverbindung wurde noch nicht initialisiert")
        return self._engine
        
    def test_connection(self) -> bool:
        """Testet die Datenbankverbindung"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Verbindungstest fehlgeschlagen: {str(e)}")
            return False 