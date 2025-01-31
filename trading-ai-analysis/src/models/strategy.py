import backtrader as bt
import numpy as np
from .risk_management import RiskManager
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from utils.ai_logger import AILogger
from utils.console_logger import ConsoleLogger

class MeanReversionStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('devfactor', 2),
        ('risk_manager', None),
    )
    
    def __init__(self):
        super().__init__()
        
        # Indikatoren
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.period)
        self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.p.period)
        self.zscore = (self.data.close - self.sma) / self.stddev
        
        # Trading-Status
        self.order = None
        self.position_size = 0
        
        self.logger = AILogger(name="mean_reversion_strategy")
        self.console = ConsoleLogger(name="mean_reversion")
        
        # Konvertiere Daten in Pandas DataFrame für ta-lib
        self.df = bt.feeds.PandasData(dataname=self.data0)
        
        # Technische Indikatoren
        self.bb = BollingerBands(
            close=self.df['close'],
            window=self.params.period,
            window_dev=self.params.devfactor
        )
        self.rsi = RSIIndicator(
            close=self.df['close'],
            window=14
        )
        self.atr = AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=14
        )
        
        # Initiales Logging der Strategie-Parameter
        self.logger.log_model_metrics(
            model_name="mean_reversion",
            metrics={
                "period": self.p.period,
                "devfactor": self.p.devfactor
            }
        )
        
        # Konsolen-Ausgabe der Strategie-Initialisierung
        self.console.section("Mean Reversion Strategie")
        self.console.info("Parameter:")
        self.console.info(f"  - Period: {self.p.period}")
        self.console.info(f"  - Devfactor: {self.p.devfactor}")

    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.zscore < -self.p.devfactor:
                size = self.get_position_size()
                self.order = self.buy(size=size)
                
        else:
            if self.zscore > 0:
                self.order = self.sell(size=self.position.size)
                
    def get_position_size(self):
        if self.p.risk_manager:
            portfolio_value = self.broker.getvalue()
            risk_adjusted_size = self.p.risk_manager.calculate_position_size(
                symbol=self.data._name,
                price=self.data.close[0],
                volatility=self.stddev[0]
            )
            return min(risk_adjusted_size, portfolio_value * 0.1)  # Max 10% des Portfolios
        return 1  # Standard-Größe wenn kein Risikomanager verfügbar
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def _buy_signal(self):
        signal = (
            self.data.close[0] < self.bb.bollinger_lband()[-1] and
            self.rsi.rsi()[-1] < 30 and
            self.data.sentiment_score > 0.5
        )
        
        if signal:
            self.logger.log_prediction(
                symbol=self.data._name,
                prediction=1.0,  # 1.0 für Kauf-Signal
                confidence=min(1.0, (30 - self.rsi.rsi()[-1]) / 30),  # Konfidenz basierend auf RSI
                features={
                    "price": self.data.close[0],
                    "bb_lower": self.bb.bollinger_lband()[-1],
                    "rsi": self.rsi.rsi()[-1],
                    "sentiment": self.data.sentiment_score
                }
            )
            
            # Konsolen-Ausgabe des Kauf-Signals
            self.console.info("\nKauf-Signal erkannt:")
            self.console.info(f"  Preis unter BB: {self.data.close[0]:.2f} < {self.bb.bollinger_lband()[-1]:.2f}")
            self.console.info(f"  RSI überverkauft: {self.rsi.rsi()[-1]:.2f}")
            self.console.info(f"  Sentiment positiv: {self.data.sentiment_score:.2f}")
        
        return signal

    def _sell_signal(self):
        signal = (
            self.data.close[0] > self.bb.bollinger_hband()[-1] or
            self.rsi.rsi()[-1] > 70 or
            self.data.close[0] < (self.data.close[0] - 2 * self.atr.average_true_range()[-1])
        )
        
        if signal:
            self.logger.log_prediction(
                symbol=self.data._name,
                prediction=-1.0,  # -1.0 für Verkauf-Signal
                confidence=min(1.0, (self.rsi.rsi()[-1] - 70) / 30),  # Konfidenz basierend auf RSI
                features={
                    "price": self.data.close[0],
                    "bb_upper": self.bb.bollinger_hband()[-1],
                    "rsi": self.rsi.rsi()[-1],
                    "atr": self.atr.average_true_range()[-1]
                }
            )
            
            # Konsolen-Ausgabe des Verkauf-Signals
            self.console.info("\nVerkauf-Signal erkannt:")
            if self.data.close[0] > self.bb.bollinger_hband()[-1]:
                self.console.info(f"  Preis über BB: {self.data.close[0]:.2f} > {self.bb.bollinger_hband()[-1]:.2f}")
            if self.rsi.rsi()[-1] > 70:
                self.console.info(f"  RSI überkauft: {self.rsi.rsi()[-1]:.2f}")
            if self.data.close[0] < (self.data.close[0] - 2 * self.atr.average_true_range()[-1]):
                self.console.info(f"  Stop-Loss ausgelöst (2 ATR)")
        
        return signal

    def _calculate_stop_loss(self):
        volatility = np.std([self.data.close[i] for i in range(-20, 0)])
        return self.data.close[0] * (1 - self.p.risk_manager.dynamic_stop_loss(
            volatility, 
            self.data.sentiment_score[0]
        ))

    def _calculate_take_profit(self):
        return self.data.close[0] * (1 + 2 * self.p.risk_manager.dynamic_stop_loss(
            np.std([self.data.close[i] for i in range(-20, 0)]),
            self.data.sentiment_score[0]
        ))

    def _current_exposure(self):
        return {self.data._name: {
            'exposure': self.position.size * self.data.close[0],
            'sector': 'technology'  # Beispiel, würde normalerweise aus DB kommen
        }} 