from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
from ..utils.logger import setup_logger
import numpy as np
from dataclasses import dataclass

logger = setup_logger(__name__)

@dataclass
class RisikoGrenzen:
    max_position_size: float = 0.05  # Maximale Größe einer einzelnen Position (5% des Portfolios)
    max_sektor_exposure: float = 0.20  # Maximale Exposure pro Sektor (20%)
    max_drawdown: float = 0.10  # Maximaler erlaubter Drawdown (10%)
    stop_loss_minimum: float = 0.10  # Minimaler Stop-Loss (10%)
    max_leverage: float = 1.2  # Maximaler Hebel
    min_diversifikation: int = 10  # Minimale Anzahl verschiedener Positionen
    max_korrelation: float = 0.6  # Maximale Korrelation zwischen Positionen
    liquiditaets_reserve: float = 0.15  # Mindest-Liquiditätsreserve (15%)

class RiskManager:
    def __init__(self, config_file: str = "risk_config.json"):
        self.config_file = Path(config_file)
        self.risiko_grenzen = self._load_config()
        self.position_history: List[Dict[str, Any]] = []
        self.circuit_breaker = False
        self.error_count = 0
        
    def _load_config(self) -> RisikoGrenzen:
        """Lädt die Risikomanagement-Konfiguration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return RisikoGrenzen(**config)
            return RisikoGrenzen()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Risikokonfiguration: {str(e)}")
            return RisikoGrenzen()

    def _save_config(self) -> None:
        """Speichert die aktuelle Risikomanagement-Konfiguration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(vars(self.risiko_grenzen), f, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Risikokonfiguration: {str(e)}")

    def validate_position(self, position: Dict[str, Any], portfolio_value: float) -> Dict[str, Any]:
        """
        Überprüft eine Position auf Einhaltung der Risikoparameter
        """
        validation_result = {
            "position_id": position.get("position_id"),
            "ist_valid": True,
            "warnungen": [],
            "anpassungen": []
        }

        # Prüfe Positionsgröße
        position_size = float(position.get("position_size", 0))
        max_position_value = portfolio_value * self.risiko_grenzen.max_position_size
        
        if position_size > max_position_value:
            validation_result["ist_valid"] = False
            validation_result["warnungen"].append(
                f"Position überschreitet maximale Größe von {self.risiko_grenzen.max_position_size*100}%"
            )
            validation_result["anpassungen"].append({
                "typ": "reduce_position",
                "ziel_groesse": max_position_value
            })

        # Prüfe Stop-Loss
        stop_loss = position.get("stop_loss")
        if stop_loss:
            stop_loss_percent = abs(float(stop_loss) - float(position.get("entry_price", 0))) / float(position.get("entry_price", 0))
            if stop_loss_percent < self.risiko_grenzen.stop_loss_minimum:
                validation_result["warnungen"].append(
                    f"Stop-Loss zu eng gesetzt (minimum {self.risiko_grenzen.stop_loss_minimum*100}%)"
                )
                validation_result["anpassungen"].append({
                    "typ": "adjust_stop_loss",
                    "minimum_abstand": self.risiko_grenzen.stop_loss_minimum
                })

        return validation_result

    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]], portfolio_value: float) -> Dict[str, Any]:
        """
        Berechnet das Gesamtrisiko des Portfolios
        """
        risk_assessment = {
            "gesamt_risiko": 0.0,
            "sektor_exposure": {},
            "diversifikation_score": 0.0,
            "liquiditaet": 0.0,
            "warnungen": [],
            "empfehlungen": []
        }

        # Berechne Sektor-Exposure
        total_exposure = 0.0
        for position in positions:
            sektor = position.get("sektor", "Unbekannt")
            position_value = float(position.get("position_size", 0))
            total_exposure += position_value
            
            if sektor in risk_assessment["sektor_exposure"]:
                risk_assessment["sektor_exposure"][sektor] += position_value
            else:
                risk_assessment["sektor_exposure"][sektor] = position_value

        # Prüfe Sektor-Limits
        for sektor, exposure in risk_assessment["sektor_exposure"].items():
            exposure_percent = exposure / portfolio_value
            if exposure_percent > self.risiko_grenzen.max_sektor_exposure:
                risk_assessment["warnungen"].append(
                    f"Zu hohe Exposure im Sektor {sektor} ({exposure_percent*100:.1f}%)"
                )
                risk_assessment["empfehlungen"].append({
                    "typ": "reduce_sector",
                    "sektor": sektor,
                    "ziel_exposure": portfolio_value * self.risiko_grenzen.max_sektor_exposure
                })

        # Berechne Diversifikation
        num_positions = len(positions)
        if num_positions < self.risiko_grenzen.min_diversifikation:
            risk_assessment["warnungen"].append(
                f"Zu wenig Diversifikation (minimum {self.risiko_grenzen.min_diversifikation} Positionen)"
            )
            risk_assessment["empfehlungen"].append({
                "typ": "increase_diversification",
                "ziel_positionen": self.risiko_grenzen.min_diversifikation
            })

        # Berechne Liquidität
        liquiditaet = portfolio_value - total_exposure
        liquiditaets_quote = liquiditaet / portfolio_value
        if liquiditaets_quote < self.risiko_grenzen.liquiditaets_reserve:
            risk_assessment["warnungen"].append(
                f"Liquiditätsreserve zu niedrig ({liquiditaets_quote*100:.1f}%)"
            )
            risk_assessment["empfehlungen"].append({
                "typ": "increase_liquidity",
                "ziel_liquiditaet": portfolio_value * self.risiko_grenzen.liquiditaets_reserve
            })

        risk_assessment["liquiditaet"] = liquiditaets_quote
        return risk_assessment

    def calculate_position_sizing(
        self,
        portfolio_value: float,
        current_positions: List[Dict[str, Any]],
        target_position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Berechnet die optimale Positionsgröße für eine neue Position
        """
        # Berechne verfügbares Kapital
        total_invested = sum(float(pos.get("position_size", 0)) for pos in current_positions)
        available_capital = portfolio_value - total_invested
        
        # Berechne Basis-Positionsgröße
        base_position_size = min(
            portfolio_value * self.risiko_grenzen.max_position_size,
            available_capital * 0.5  # Maximal 50% des verfügbaren Kapitals
        )

        # Adjustiere basierend auf Risikofaktoren
        risk_factors = {
            "volatility": float(target_position.get("volatility", 1.0)),
            "market_risk": float(target_position.get("market_risk", 1.0)),
            "correlation": float(target_position.get("correlation", 0.5))
        }
        
        # Gewichte die Risikofaktoren
        risk_adjustment = np.mean([
            1.0 / risk_factors["volatility"],
            1.0 / risk_factors["market_risk"],
            1.0 / risk_factors["correlation"] if risk_factors["correlation"] > 0 else 2.0
        ])

        recommended_size = base_position_size * risk_adjustment

        return {
            "empfohlene_groesse": min(recommended_size, base_position_size),
            "max_groesse": base_position_size,
            "risiko_faktoren": risk_factors,
            "verfuegbares_kapital": available_capital
        }

    def update_risk_limits(self, new_limits: Dict[str, float]) -> None:
        """
        Aktualisiert die Risikolimits basierend auf Performance und Marktbedingungen
        """
        try:
            for key, value in new_limits.items():
                if hasattr(self.risiko_grenzen, key):
                    setattr(self.risiko_grenzen, key, value)
            self._save_config()
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Risikolimits: {str(e)}")

    def analyze_drawdown(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analysiert den Drawdown und gibt Warnungen aus
        """
        if not portfolio_history:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0, "warnungen": []}

        values = [entry["portfolio_value"] for entry in portfolio_history]
        peak = values[0]
        max_drawdown = 0.0
        current_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            if value == values[-1]:
                current_drawdown = drawdown

        result = {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "warnungen": []
        }

        if current_drawdown > self.risiko_grenzen.max_drawdown:
            result["warnungen"].append({
                "typ": "drawdown_limit",
                "message": f"Drawdown ({current_drawdown*100:.1f}%) überschreitet Limit von {self.risiko_grenzen.max_drawdown*100}%",
                "empfehlung": "Risikoreduktion erforderlich"
            })

        return result

    def _emergency_shutdown(self):
        """Notfall-Prozedur bei kritischen Fehlern"""
        logger.critical("Aktiviere Notabschaltung!")
        self.circuit_breaker = True
        # Schließe alle Positionen
        # Stoppe alle laufenden Prozesse
        # Sichere kritische Daten 