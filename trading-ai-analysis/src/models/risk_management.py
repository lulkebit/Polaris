import numpy as np
from typing import Dict
from storage.database import DatabaseConnection
from utils.ai_logger import AILogger

class RiskManager:
    def __init__(self, initial_capital: float = 100000.0):
        self.db = DatabaseConnection()
        self.initial_capital = initial_capital
        self.risk_parameters = self._load_risk_parameters()
        self.logger = AILogger(name="risk_manager")
        
        # Log initial risk parameters
        self.logger.log_model_metrics(
            model_name="risk_manager",
            metrics={
                "initial_capital": self.initial_capital,
                **self.risk_parameters
            }
        )
        
    def _load_risk_parameters(self) -> Dict:
        """Lädt Risikoparameter aus der Datenbank"""
        return {
            'max_portfolio_risk': 0.02,  # 2% des Gesamtkapitals
            'max_position_risk': 0.01,   # 1% pro Position
            'max_leverage': 3,
            'stop_loss_pct': 0.05,
            'volatility_adjustment': True,
            'sector_limits': {
                'technology': 0.3,
                'finance': 0.2,
                'energy': 0.15
            }
        }

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Berechnet die Positionsgröße basierend auf Risikoparametern"""
        risk_per_share = entry_price - stop_loss
        max_risk_amount = self.initial_capital * self.risk_parameters['max_position_risk']
        position_size = min(max_risk_amount / risk_per_share, 
                          self.initial_capital * self.risk_parameters['max_leverage'] / entry_price)
        
        # Log position sizing calculation
        self.logger.log_model_metrics(
            model_name="position_sizing",
            metrics={
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "risk_per_share": risk_per_share,
                "max_risk_amount": max_risk_amount,
                "position_size": position_size
            }
        )
        
        return position_size

    def dynamic_stop_loss(self, volatility: float, sentiment: float) -> float:
        """Berechnet dynamischen Stop-Loss basierend auf Volatilität und Sentiment"""
        base_sl = self.risk_parameters['stop_loss_pct']
        volatility_factor = np.tanh(volatility * 10)  # Skaliert Volatilität auf 0-1
        sentiment_adjustment = 0.5 + (0.5 - sentiment)  # Sentiment 0-1 skaliert
        stop_loss = base_sl * (1 + volatility_factor) * sentiment_adjustment
        
        # Log stop loss calculation
        self.logger.log_model_metrics(
            model_name="dynamic_stop_loss",
            metrics={
                "volatility": volatility,
                "sentiment": sentiment,
                "volatility_factor": volatility_factor,
                "sentiment_adjustment": sentiment_adjustment,
                "final_stop_loss": stop_loss
            }
        )
        
        return stop_loss

    def portfolio_risk_check(self, current_positions: dict) -> bool:
        """Überprüft Portfolio-Risikolimits"""
        total_exposure = sum(pos['exposure'] for pos in current_positions.values())
        sector_exposure = self._calculate_sector_exposure(current_positions)
        
        risk_ok = (
            total_exposure <= self.initial_capital * self.risk_parameters['max_leverage'] and
            all(v <= self.risk_parameters['sector_limits'][k] 
              for k, v in sector_exposure.items())
        )
        
        # Log risk check results
        self.logger.log_model_metrics(
            model_name="portfolio_risk_check",
            metrics={
                "total_exposure": total_exposure,
                "max_allowed_exposure": self.initial_capital * self.risk_parameters['max_leverage'],
                "risk_check_passed": risk_ok,
                **{"sector_exposure_" + k: v for k, v in sector_exposure.items()}
            }
        )
        
        return risk_ok

    def _calculate_sector_exposure(self, positions: dict) -> Dict[str, float]:
        """Berechnet die Sektor-Exposure-Verteilung"""
        sectors = {'technology': 0.4, 'finance': 0.3, 'energy': 0.3}  # Beispiel
        exposures = {s: sum(p['exposure'] for sym, p in positions.items() 
                        if sectors.get(sym) == s) / self.initial_capital
                    for s in self.risk_parameters['sector_limits']}
        
        # Log sector exposures
        self.logger.log_model_metrics(
            model_name="sector_exposure",
            metrics=exposures
        )
        
        return exposures 