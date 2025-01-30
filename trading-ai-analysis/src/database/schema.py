from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey, Enum, Boolean, CheckConstraint, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class TradeType(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class TradeStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey('backtest_results.id'))
    symbol = Column(String(20), nullable=False)
    trade_type = Column(Enum(TradeType), nullable=False)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float)
    commission = Column(Float)
    slippage = Column(Float)
    strategy_parameters = Column(JSON)
    meta_info = Column(JSON)

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    trades = relationship("Trade", backref="backtest")
    strategy_config = Column(JSON)
    performance_metrics = Column(JSON)
    risk_metrics = Column(JSON)
    trading_signals = Column(JSON)
    meta_info = Column(JSON)

class MarketAnalysis(Base):
    __tablename__ = 'market_analyses'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    technical_indicators = Column(JSON)
    sentiment_analysis = Column(JSON)
    trend_analysis = Column(JSON)
    risk_assessment = Column(JSON)
    trading_signals = Column(JSON)
    meta_info = Column(JSON)

class OptimizationResult(Base):
    __tablename__ = 'optimization_results'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    strategy_name = Column(String(100))
    initial_parameters = Column(JSON)
    optimized_parameters = Column(JSON)
    performance_improvement = Column(Float)
    optimization_metrics = Column(JSON)
    backtest_results_id = Column(Integer, ForeignKey('backtest_results.id'))
    meta_info = Column(JSON)

class TradeDirection(enum.Enum):
    LONG = "long"
    SHORT = "short"

class SignalStatus(enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class TradingAIAnalysis(Base):
    """Haupttabelle für Analyseergebnisse"""
    __tablename__ = "trading_ai_analyses"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    market_conditions = Column(JSON, nullable=False)
    performance_metrics = Column(JSON, nullable=False)
    risk_metrics = Column(JSON, nullable=False)
    meta_info = Column(JSON, nullable=True)
    
    # Beziehungen
    trade_signals = relationship("TradeSignal", back_populates="analysis")
    market_data = relationship("MarketData", back_populates="analysis")

class TradeSignal(Base):
    """Tabelle für Handelssignale"""
    __tablename__ = "trade_signals"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("trading_ai_analyses.id"))
    symbol = Column(String(20), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    signal_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    expiry_time = Column(DateTime, nullable=True)
    status = Column(Enum(SignalStatus), nullable=False, default=SignalStatus.PENDING)
    meta_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Beziehungen
    analysis = relationship("TradingAIAnalysis", back_populates="trade_signals")

class MarketData(Base):
    """Tabelle für Marktdaten"""
    __tablename__ = "market_data_combined"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_normalized = Column(Float, nullable=True)
    analysis_id = Column(Integer, ForeignKey('trading_ai_analyses.id'), nullable=True)
    
    # Beziehung
    analysis = relationship("TradingAIAnalysis", back_populates="market_data")

class NewsData(Base):
    """Tabelle für Nachrichtendaten"""
    __tablename__ = "news_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=False)
    url = Column(String(500), nullable=True)
    sentiment = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

class PerformanceMetrics(Base):
    """Tabelle für Performance-Metriken"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("trading_ai_analyses.id"))
    timestamp = Column(DateTime, nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    meta_info = Column(JSON, nullable=True)
    
    # Beziehung
    analysis = relationship("TradingAIAnalysis")

class RiskMetrics(Base):
    """Tabelle für Risiko-Metriken"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("trading_ai_analyses.id"))
    timestamp = Column(DateTime, nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    meta_info = Column(JSON, nullable=True)
    
    # Beziehung
    analysis = relationship("TradingAIAnalysis")

def init_db(engine):
    """Initialisiert die Datenbankstruktur"""
    Base.metadata.create_all(engine) 