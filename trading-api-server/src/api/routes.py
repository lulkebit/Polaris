from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from database.database import get_db
from models.analysis import AIAnalysis
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/analyses/latest")
async def get_latest_analyses(limit: int = 10, db: Session = Depends(get_db)):
    """
    Holt die neuesten AI-Analysen aus der Datenbank
    """
    try:
        analyses = (
            db.query(AIAnalysis)
            .order_by(AIAnalysis.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        if not analyses:
            logger.warning("Keine Analysen in der Datenbank gefunden")
            return []
            
        logger.info(f"Erfolgreich {len(analyses)} Analysen abgerufen")
        return analyses
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Analysen: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Fehler beim Abrufen der Analysen aus der Datenbank"
        )

@router.get("/analyses/by-date/{date}")
async def get_analyses_by_date(date: str, db: Session = Depends(get_db)):
    """
    Holt Analysen für ein bestimmtes Datum
    """
    try:
        # Konvertiere das Datum in das erwartete Format
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        date_str = date_obj.strftime("%Y-%m-%d")
        next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        
        analyses = (
            db.query(AIAnalysis)
            .filter(AIAnalysis.timestamp >= date_str)
            .filter(AIAnalysis.timestamp < next_day)
            .order_by(AIAnalysis.timestamp.desc())
            .all()
        )
        
        if not analyses:
            logger.warning(f"Keine Analysen für das Datum {date} gefunden")
            return []
            
        logger.info(f"Erfolgreich {len(analyses)} Analysen für {date} abgerufen")
        return analyses
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Ungültiges Datumsformat. Bitte YYYY-MM-DD verwenden"
        )
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Analysen für Datum {date}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Fehler beim Abrufen der Analysen aus der Datenbank"
        )

@router.get("/market-analysis/{analysis_id}")
async def get_market_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """
    Holt eine spezifische Marktanalyse
    """
    try:
        analysis = db.query(MarketAnalysis).filter(MarketAnalysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analyse nicht gefunden")
        logger.info(f"Marktanalyse {analysis_id} erfolgreich abgerufen")
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Marktanalyse {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Interner Serverfehler")

@router.get("/news-analysis/{analysis_id}")
async def get_news_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """
    Holt eine spezifische Nachrichtenanalyse
    """
    try:
        analysis = db.query(NewsAnalysis).filter(NewsAnalysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analyse nicht gefunden")
        logger.info(f"Nachrichtenanalyse {analysis_id} erfolgreich abgerufen")
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Nachrichtenanalyse {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Interner Serverfehler") 