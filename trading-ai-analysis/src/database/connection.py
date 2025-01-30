import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from dotenv import load_dotenv
import time
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from .schema import Base
from config.settings import DATABASE_CONFIG
from utils.logging_config import get_logger

class DatabaseConnection:
    """Verwaltet Datenbankverbindungen mit Connection Pooling"""
    
    def __init__(self):
        self.logger = get_logger()
        self._engine = None
        self._session_factory = None
        self.initialize()
    
    def initialize(self) -> None:
        """Initialisiert die Datenbankverbindung"""
        try:
            # Erstelle Connection String
            db_url = (
                f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
                f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
                f"/{DATABASE_CONFIG['database']}"
            )
            
            # Konfiguriere Connection Pool
            self._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle Connections nach 30 Minuten
                echo=False
            )
            
            # Erstelle Session Factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            self.logger.logger.info(
                "Datenbankverbindung initialisiert",
                extra={"host": DATABASE_CONFIG['host'], "database": DATABASE_CONFIG['database']}
            )
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler bei der Initialisierung der Datenbankverbindung: {str(e)}",
                exc_info=True
            )
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context Manager für Datenbanksessions mit automatischer Fehlerbehandlung"""
        session = self._session_factory()
        max_retries = 3
        retry_count = 0
        
        try:
            yield session
            
        except OperationalError as e:
            # Versuche bei Verbindungsproblemen einen Retry
            while retry_count < max_retries:
                try:
                    retry_count += 1
                    time.sleep(1)  # Warte kurz vor dem Retry
                    
                    # Schließe alte Session und erstelle neue
                    session.close()
                    session = self._session_factory()
                    
                    yield session
                    break
                    
                except OperationalError as retry_error:
                    if retry_count == max_retries:
                        self.logger.logger.error(
                            f"Maximale Anzahl an Retries erreicht: {str(retry_error)}",
                            exc_info=True
                        )
                        raise
                    
                    self.logger.logger.warning(
                        f"Datenbankverbindung fehlgeschlagen, Retry {retry_count}/{max_retries}"
                    )
            
        except SQLAlchemyError as e:
            self.logger.logger.error(f"Datenbankfehler: {str(e)}", exc_info=True)
            session.rollback()
            raise
            
        except Exception as e:
            self.logger.logger.error(f"Unerwarteter Fehler: {str(e)}", exc_info=True)
            session.rollback()
            raise
            
        finally:
            session.close()
    
    def close(self) -> None:
        """Schließt alle Datenbankverbindungen"""
        if self._engine:
            self._engine.dispose()
            self.logger.logger.info("Datenbankverbindungen geschlossen")
    
    def check_connection(self) -> bool:
        """Überprüft die Datenbankverbindung"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            self.logger.logger.error(
                f"Verbindungstest fehlgeschlagen: {str(e)}",
                exc_info=True
            )
            return False 