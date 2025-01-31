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
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('rsi_period', 14),
        ('atr_period', 14),
        ('risk_manager', None)  # Risikomanager-Instanz
    )

    def __init__(self):
        super().__init__()
        self.risk_manager = self.p.risk_manager or RiskManager()
        self.logger = AILogger(name="mean_reversion_strategy")
        self.console = ConsoleLogger(name="mean_reversion")
        
        # Konvertiere Daten in Pandas DataFrame für ta-lib
        self.df = bt.feeds.PandasData(dataname=self.data0)
        
        # Technische Indikatoren
        self.bb = BollingerBands(
            close=self.df['close'],
            window=self.params.bb_period,
            window_dev=self.params.bb_dev
        )
        self.rsi = RSIIndicator(
            close=self.df['close'],
            window=self.params.rsi_period
        )
        self.atr = AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=self.params.atr_period
        )
        
        # Initiales Logging der Strategie-Parameter
        self.logger.log_model_metrics(
            model_name="mean_reversion",
            metrics={
                "bb_period": self.p.bb_period,
                "bb_dev": self.p.bb_dev,
                "rsi_period": self.p.rsi_period,
                "atr_period": self.p.atr_period
            }
        )
        
        # Konsolen-Ausgabe der Strategie-Initialisierung
        self.console.section("Mean Reversion Strategie")
        self.console.info("Parameter:")
        self.console.info(f"  - Bollinger Band Periode: {self.p.bb_period}")
        self.console.info(f"  - Bollinger Band Deviation: {self.p.bb_dev}")
        self.console.info(f"  - RSI Periode: {self.p.rsi_period}")
        self.console.info(f"  - ATR Periode: {self.p.atr_period}")

    def next(self):
        # Aktuelle Indikatoren loggen
        current_indicators = {
            "bb_upper": self.bb.bollinger_hband()[-1],
            "bb_lower": self.bb.bollinger_lband()[-1],
            "bb_middle": self.bb.bollinger_mavg()[-1],
            "rsi": self.rsi.rsi()[-1],
            "atr": self.atr.average_true_range()[-1]
        }
        
        self.logger.log_indicators(
            symbol=self.data._name,
            indicators=current_indicators
        )
        
        # Konsolen-Ausgabe der aktuellen Indikatoren
        self.console.info(f"\nAnalyse für {self.data._name}:")
        self.console.info(f"  Preis: {self.data.close[0]:.2f}")
        self.console.info(f"  RSI: {current_indicators['rsi']:.2f}")
        self.console.info(f"  BB Bänder: {current_indicators['bb_lower']:.2f} - {current_indicators['bb_middle']:.2f} - {current_indicators['bb_upper']:.2f}")

        if not self.position:
            if self._buy_signal():
                sl = self._calculate_stop_loss()
                tp = self._calculate_take_profit()
                size = self.risk_manager.calculate_position_size(
                    self.data.close[0], 
                    sl
                )
                if self.risk_manager.portfolio_risk_check(self._current_exposure()):
                    self.buy(size=size, exectype=bt.Order.Limit, price=self.data.close[0])
                    self.sell(size=size, exectype=bt.Order.Stop, price=sl)
                    self.sell(size=size, exectype=bt.Order.Limit, price=tp)
                    
                    # Trade loggen
                    self.logger.log_trade(
                        symbol=self.data._name,
                        action="buy",
                        price=self.data.close[0],
                        size=size,
                        reason="RSI oversold + BB breakout",
                        risk_metrics={
                            "stop_loss": sl,
                            "take_profit": tp,
                            "risk_per_trade": (self.data.close[0] - sl) * size,
                            "reward_risk_ratio": (tp - self.data.close[0]) / (self.data.close[0] - sl)
                        }
                    )
                    
                    # Konsolen-Ausgabe des Trades
                    self.console.section(f"Neuer Trade: KAUF {self.data._name}")
                    self.console.success(f"Kauf ausgeführt:")
                    self.console.info(f"  Preis: {self.data.close[0]:.2f}")
                    self.console.info(f"  Größe: {size:.4f}")
                    self.console.info(f"  Stop Loss: {sl:.2f}")
                    self.console.info(f"  Take Profit: {tp:.2f}")
                    self.console.info(f"  R/R Ratio: {(tp - self.data.close[0]) / (self.data.close[0] - sl):.2f}")
        else:
            if self._sell_signal():
                self.close()
                # Trade loggen
                self.logger.log_trade(
                    symbol=self.data._name,
                    action="sell",
                    price=self.data.close[0],
                    size=self.position.size,
                    reason="Signal conditions met",
                    risk_metrics={
                        "profit_loss": (self.data.close[0] - self.position.price) * self.position.size
                    }
                )
                
                # Konsolen-Ausgabe des Verkaufs
                profit_loss = (self.data.close[0] - self.position.price) * self.position.size
                if profit_loss > 0:
                    self.console.success(f"Position geschlossen mit Gewinn: {profit_loss:.2f}")
                else:
                    self.console.failure(f"Position geschlossen mit Verlust: {profit_loss:.2f}")

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
        return self.data.close[0] * (1 - self.risk_manager.dynamic_stop_loss(
            volatility, 
            self.data.sentiment_score[0]
        ))

    def _calculate_take_profit(self):
        return self.data.close[0] * (1 + 2 * self.risk_manager.dynamic_stop_loss(
            np.std([self.data.close[i] for i in range(-20, 0)]),
            self.data.sentiment_score[0]
        ))

    def _current_exposure(self):
        return {self.data._name: {
            'exposure': self.position.size * self.data.close[0],
            'sector': 'technology'  # Beispiel, würde normalerweise aus DB kommen
        }} 