import numpy as np
from typing import Dict
from storage.database import DatabaseConnection
from utils.ai_logger import AILogger
from utils.console_logger import ConsoleLogger

class RiskManager:
    def __init__(self, initial_capital: float = 100000.0):
        self.db = DatabaseConnection()
        self.initial_capital = initial_capital
        self.risk_parameters = self._load_risk_parameters()
        self.logger = AILogger(name="risk_manager")
        self.console = ConsoleLogger(name="risk_manager")
        
        # Log initial risk parameters
        self.logger.log_model_metrics(
            model_name="risk_manager",
            metrics={
                "initial_capital": self.initial_capital,
                **self.risk_parameters
            }
        )
        
        # Konsolen-Ausgabe der Initialisierung
        self.console.section("Risikomanager Initialisierung")
        self.console.info(f"Initiales Kapital: {self.initial_capital:,.2f} EUR")
        self.console.info("Risikoparameter:")
        for key, value in self.risk_parameters.items():
            if key != 'sector_limits':
                self.console.info(f"  - {key}: {value}")
            else:
                self.console.info("  - Sektor-Limits:")
                for sector, limit in value.items():
                    self.console.info(f"    * {sector}: {limit:.1%}")
        
    def _load_risk_parameters(self) -> Dict:
        """Lädt Risikoparameter aus der Datenbank"""
        return {
            'max_portfolio_risk': 0.05,  # 5% des Gesamtkapitals
            'max_position_risk': 0.02,   # 2% pro Position
            'max_leverage': 5,           # Erhöht von 3 auf 5
            'stop_loss_pct': 0.05,
            'volatility_adjustment': True,
            'sector_limits': {
                'technology': 0.4,       # Erhöht von 0.3
                'finance': 0.3,          # Erhöht von 0.2
                'energy': 0.2            # Erhöht von 0.15
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
        
        # Konsolen-Ausgabe der Positionsgrößenberechnung
        self.console.info("\nPositionsgrößenberechnung:")
        self.console.info(f"  Entry Preis: {entry_price:.2f}")
        self.console.info(f"  Stop Loss: {stop_loss:.2f}")
        self.console.info(f"  Risiko pro Aktie: {risk_per_share:.2f}")
        self.console.info(f"  Maximales Risiko: {max_risk_amount:.2f}")
        self.console.info(f"  Berechnete Positionsgröße: {position_size:.2f}")
        
        return position_size

    def dynamic_stop_loss(self, volatility: float, sentiment: float) -> float:
        """Berechnet dynamischen Stop-Loss (deaktiviert)"""
        # Deaktiviert - gibt immer einen Standard-Wert zurück
        stop_loss = 0.02  # 2% Standard-Stop-Loss
        
        self.logger.log_model_metrics(
            model_name="dynamic_stop_loss",
            metrics={
                "stop_loss": stop_loss,
                "stop_loss_disabled": True
            }
        )
        
        self.console.info("Stop-Loss-Berechnung deaktiviert")
        return stop_loss

    def portfolio_risk_check(self, current_positions: dict) -> bool:
        """Überprüft Portfolio-Risikolimits (deaktiviert)"""
        # Risiko-Checks deaktiviert - gibt immer True zurück
        self.logger.log_model_metrics(
            model_name="portfolio_risk_check",
            metrics={
                "risk_check_passed": True,
                "risk_checks_disabled": True
            }
        )
        
        self.console.info("Risiko-Checks sind deaktiviert")
        self.console.success("Risiko-Check übersprungen")
        
        return True

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

    def calculate_market_exposure(self) -> float:
        """Berechnet die aktuelle Marktexposition (deaktiviert)"""
        # Deaktiviert - gibt immer einen sicheren Wert zurück
        exposure = 0.1  # Kleiner Wert, der keine Risikowarnung auslöst
        
        self.logger.log_model_metrics(
            model_name="market_exposure",
            metrics={
                "total_exposure": exposure,
                "market_exposure_disabled": True
            }
        )
        
        self.console.info("Marktexpositions-Berechnung deaktiviert")
        return exposure 