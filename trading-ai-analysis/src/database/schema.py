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
    meta_data = Column(JSON)

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
    meta_data = Column(JSON)

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
    meta_data = Column(JSON)

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
    meta_data = Column(JSON)

class TradingAIAnalysis(Base):
    __tablename__ = 'trading_ai_analyses'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    analysis_type = Column(String(50), nullable=False)
    market_data_analysis = Column(JSON)
    news_analysis = Column(JSON)
    position_analysis = Column(JSON)
    historical_analysis = Column(JSON)
    recommendations = Column(JSON)
    confidence_score = Column(Integer)
    
    # Validierung für JSON-Felder
    __table_args__ = (
        CheckConstraint(
            "json_typeof(market_data_analysis) = 'object'",
            name='check_market_data_analysis_json'
        ),
        CheckConstraint(
            "json_typeof(news_analysis) = 'object'",
            name='check_news_analysis_json'
        ),
        CheckConstraint(
            "json_typeof(position_analysis) = 'object'",
            name='check_position_analysis_json'
        ),
        CheckConstraint(
            "json_typeof(historical_analysis) = 'object'",
            name='check_historical_analysis_json'
        ),
        CheckConstraint(
            "json_typeof(recommendations) = 'object'",
            name='check_recommendations_json'
        ),
        CheckConstraint(
            "confidence_score BETWEEN 0 AND 100",
            name='check_confidence_score_range'
        ),
    )

    def __repr__(self):
        return f"<TradingAIAnalysis(id={self.id}, timestamp={self.timestamp}, type={self.analysis_type})>"

class MarketData(Base):
    __tablename__ = 'market_data_combined'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_normalized = Column(Float)
    
    __table_args__ = (
        CheckConstraint('open > 0'),
        CheckConstraint('high > 0'),
        CheckConstraint('low > 0'),
        CheckConstraint('close > 0'),
        CheckConstraint('volume >= 0'),
    )

class NewsData(Base):
    __tablename__ = 'news'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    publishedAt = Column(DateTime)
    url = Column(String(1000))
    sentiment = Column(Float)  # -1 bis 1

def init_db(engine):
    """Initialisiert die Datenbankstruktur"""
    Base.metadata.create_all(engine) 