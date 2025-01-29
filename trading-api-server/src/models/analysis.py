from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from database.database import Base
from datetime import datetime

class MarketAnalysis(Base):
    __tablename__ = "market_analysis"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    market_data = Column(String)
    analysis_result = Column(String)
    confidence_score = Column(Float)
    recommendations = Column(JSON)

class NewsAnalysis(Base):
    __tablename__ = "news_analysis"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    news_title = Column(String)
    news_content = Column(String)
    sentiment_score = Column(Float)
    impact_analysis = Column(String)

class CombinedAnalysis(Base):
    __tablename__ = "combined_analysis"

    id = Column(Integer, primary_key=True, index=True)
    market_analysis_id = Column(Integer, ForeignKey("market_analysis.id"))
    news_analysis_id = Column(Integer, ForeignKey("news_analysis.id"))
    overall_recommendation = Column(String)
    risk_level = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    market_analysis = relationship("MarketAnalysis")
    news_analysis = relationship("NewsAnalysis")

class AIAnalysis(Base):
    __tablename__ = "ai_analysis"
    __table_args__ = {'extend_existing': True}

    timestamp = Column(DateTime, primary_key=True)
    analysis = Column(String)

    def __repr__(self):
        return f"<AIAnalysis(timestamp={self.timestamp}, analysis={self.analysis[:50]}...)>" 