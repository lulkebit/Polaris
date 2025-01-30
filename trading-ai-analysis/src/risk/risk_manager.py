from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    var_limit: float = 0.02  # Value at Risk Limit (2% täglich)
    stress_test_verlust_limit: float = 0.15  # Maximaler Verlust im Stress-Test (15%)

class RiskManager:
    def __init__(self, config_file: str = "risk_config.json"):
        self.config_file = Path(config_file)
        self.risiko_grenzen = self._load_config()
        self.position_history: List[Dict[str, Any]] = []
        self.circuit_breaker = False
        self.error_count = 0
        self.portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        
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
        """Validiert eine einzelne Position gegen die Risikoregeln"""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Prüfe Positionsgröße
        position_size = position['value'] / portfolio_value
        if position_size > self.risiko_grenzen.max_position_size:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Position zu groß: {position_size:.1%} > {self.risiko_grenzen.max_position_size:.1%}"
            )
        
        # Prüfe Stop-Loss
        if 'stop_loss' in position:
            stop_loss_percent = (position['entry_price'] - position['stop_loss']) / position['entry_price']
            if stop_loss_percent < self.risiko_grenzen.stop_loss_minimum:
                validation_result["warnings"].append(
                    f"Stop-Loss zu eng: {stop_loss_percent:.1%} < {self.risiko_grenzen.stop_loss_minimum:.1%}"
                )
        
        return validation_result

    def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, Any]], portfolio_value: float) -> Dict[str, Any]:
        """Berechnet das Gesamtrisiko des Portfolios"""
        risk_metrics = {
            "total_exposure": 0.0,
            "sector_exposure": {},
            "position_sizes": {},
            "correlation_matrix": None,
            "var_95": 0.0,
            "current_drawdown": self.current_drawdown,
            "diversification_score": 0.0,
            "risk_warnings": []
        }
        
        # Berechne Exposure
        for symbol, pos in positions.items():
            position_value = pos['value']
            risk_metrics["total_exposure"] += position_value
            risk_metrics["position_sizes"][symbol] = position_value / portfolio_value
            
            # Sektor-Exposure
            sector = pos.get('sector', 'Unknown')
            risk_metrics["sector_exposure"][sector] = risk_metrics["sector_exposure"].get(sector, 0) + \
                                                    position_value / portfolio_value
        
        # Prüfe Sektorgrenzen
        for sector, exposure in risk_metrics["sector_exposure"].items():
            if exposure > self.risiko_grenzen.max_sektor_exposure:
                risk_metrics["risk_warnings"].append(
                    f"Sektor-Exposure zu hoch für {sector}: {exposure:.1%}"
                )
        
        # Berechne Diversifikation Score
        num_positions = len(positions)
        risk_metrics["diversification_score"] = min(1.0, num_positions / self.risiko_grenzen.min_diversifikation)
        
        if num_positions < self.risiko_grenzen.min_diversifikation:
            risk_metrics["risk_warnings"].append(
                f"Zu wenig Diversifikation: {num_positions} < {self.risiko_grenzen.min_diversifikation}"
            )
        
        return risk_metrics

    def update_portfolio_value(self, new_value: float) -> None:
        """Aktualisiert den Portfoliowert und berechnet Drawdown"""
        self.portfolio_value = new_value
        
        if new_value > self.peak_value:
            self.peak_value = new_value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - new_value) / self.peak_value
            
            if self.current_drawdown > self.risiko_grenzen.max_drawdown:
                self.trigger_circuit_breaker(f"Maximaler Drawdown überschritten: {self.current_drawdown:.1%}")

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Berechnet den Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def perform_stress_test(self, positions: Dict[str, Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Führt Stress-Tests für verschiedene Marktszenarien durch"""
        stress_test_results = {
            "worst_case_loss": 0.0,
            "scenario_impacts": {},
            "risk_warnings": []
        }
        
        for scenario in scenarios:
            total_impact = 0.0
            for symbol, pos in positions.items():
                # Berechne Verlust basierend auf Szenario-Parameter
                impact = pos['value'] * scenario.get('market_change', 0)
                if symbol in scenario.get('specific_impacts', {}):
                    impact = pos['value'] * scenario['specific_impacts'][symbol]
                total_impact += impact
            
            stress_test_results["scenario_impacts"][scenario['name']] = total_impact
            
            if abs(total_impact) > self.portfolio_value * self.risiko_grenzen.stress_test_verlust_limit:
                stress_test_results["risk_warnings"].append(
                    f"Kritischer Verlust im Szenario {scenario['name']}: {total_impact:.2f}"
                )
        
        return stress_test_results

    def trigger_circuit_breaker(self, reason: str) -> None:
        """Aktiviert den Circuit Breaker"""
        self.circuit_breaker = True
        logger.warning(f"Circuit Breaker aktiviert: {reason}")
        
        # Hier können weitere Aktionen ausgelöst werden, z.B.:
        # - Alle offenen Orders stornieren
        # - Risikopositionen schließen
        # - Benachrichtigungen senden

    def reset_circuit_breaker(self) -> None:
        """Setzt den Circuit Breaker zurück"""
        self.circuit_breaker = False
        logger.info("Circuit Breaker zurückgesetzt")

    def analyze_drawdown(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert den historischen Drawdown"""
        equity_curve = portfolio_history['portfolio_value']
        rolling_max = equity_curve.expanding().max()
        drawdown = (rolling_max - equity_curve) / rolling_max
        
        return {
            "current_drawdown": self.current_drawdown,
            "max_historical_drawdown": drawdown.max(),
            "avg_drawdown": drawdown.mean(),
            "drawdown_periods": self._identify_drawdown_periods(drawdown),
            "recovery_times": self._calculate_recovery_times(drawdown)
        }

    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Identifiziert signifikante Drawdown-Perioden"""
        significant_drawdowns = []
        in_drawdown = False
        start_idx = None
        
        for idx, value in drawdown.items():
            if not in_drawdown and value < -0.05:  # Start eines signifikanten Drawdowns
                in_drawdown = True
                start_idx = idx
            elif in_drawdown and value > -0.02:  # Ende eines Drawdowns
                in_drawdown = False
                significant_drawdowns.append({
                    "start_date": start_idx,
                    "end_date": idx,
                    "max_drawdown": drawdown[start_idx:idx].min(),
                    "duration_days": (idx - start_idx).days
                })
        
        return significant_drawdowns

    def _calculate_recovery_times(self, drawdown: pd.Series) -> Dict[str, float]:
        """Berechnet durchschnittliche Erholungszeiten aus Drawdowns"""
        recovery_times = []
        in_drawdown = False
        start_idx = None
        
        for idx, value in drawdown.items():
            if not in_drawdown and value < -0.05:
                in_drawdown = True
                start_idx = idx
            elif in_drawdown and value == 0:
                recovery_times.append((idx - start_idx).days)
                in_drawdown = False
        
        return {
            "avg_recovery_days": np.mean(recovery_times) if recovery_times else 0,
            "max_recovery_days": max(recovery_times) if recovery_times else 0,
            "min_recovery_days": min(recovery_times) if recovery_times else 0
        }

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

    def _emergency_shutdown(self):
        """Notfall-Prozedur bei kritischen Fehlern"""
        logger.critical("Aktiviere Notabschaltung!")
        self.circuit_breaker = True
        # Schließe alle Positionen
        # Stoppe alle laufenden Prozesse
        # Sichere kritische Daten 