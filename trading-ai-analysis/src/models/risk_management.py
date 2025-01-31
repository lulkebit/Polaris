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
        
        # Konsolen-Ausgabe der Positionsgrößenberechnung
        self.console.info("\nPositionsgrößenberechnung:")
        self.console.info(f"  Entry Preis: {entry_price:.2f}")
        self.console.info(f"  Stop Loss: {stop_loss:.2f}")
        self.console.info(f"  Risiko pro Aktie: {risk_per_share:.2f}")
        self.console.info(f"  Maximales Risiko: {max_risk_amount:.2f}")
        self.console.info(f"  Berechnete Positionsgröße: {position_size:.2f}")
        
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
        
        # Konsolen-Ausgabe der Stop-Loss-Berechnung
        self.console.info("\nDynamische Stop-Loss Berechnung:")
        self.console.info(f"  Basis Stop-Loss: {base_sl:.1%}")
        self.console.info(f"  Volatilität: {volatility:.4f}")
        self.console.info(f"  Volatilitätsfaktor: {volatility_factor:.2f}")
        self.console.info(f"  Sentiment: {sentiment:.2f}")
        self.console.info(f"  Sentiment-Anpassung: {sentiment_adjustment:.2f}")
        self.console.info(f"  Finaler Stop-Loss: {stop_loss:.1%}")
        
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
        
        # Konsolen-Ausgabe der Risikochecks
        self.console.section("Portfolio Risiko Check")
        self.console.info(f"Gesamtexposure: {total_exposure:,.2f} EUR")
        self.console.info(f"Maximale erlaubte Exposure: {self.initial_capital * self.risk_parameters['max_leverage']:,.2f} EUR")
        self.console.info("\nSektor-Expositionen:")
        for sector, exposure in sector_exposure.items():
            limit = self.risk_parameters['sector_limits'][sector]
            status = "✓" if exposure <= limit else "✗"
            color = Fore.GREEN if exposure <= limit else Fore.RED
            self.console.info(f"  {status} {sector}: {exposure:.1%} (Limit: {limit:.1%})")
        
        if risk_ok:
            self.console.success("Risiko-Check bestanden")
        else:
            self.console.failure("Risiko-Check fehlgeschlagen")
        
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

    def calculate_market_exposure(self) -> float:
        """Berechnet die aktuelle Marktexposition"""
        query = """
            SELECT SUM(close * volume) / (SELECT COUNT(DISTINCT symbol) FROM market_data_combined) as exposure
            FROM market_data_combined
            WHERE timestamp = (SELECT MAX(timestamp) FROM market_data_combined)
        """
        try:
            result = self.db.execute_query(query)
            exposure = float(result[0][0] or 0) / self.initial_capital  # Normalisiert auf Initialkapital
            
            self.logger.log_model_metrics(
                model_name="market_exposure",
                metrics={"total_exposure": exposure}
            )
            
            return exposure
        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der Marktexposition: {str(e)}")
            return 0.0 