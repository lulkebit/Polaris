from ..models.deepseek_model import DeepseekAnalyzer
from ..utils.logger import setup_logger
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from ..risk.risk_manager import RiskManager

logger = setup_logger(__name__)

class MarketAnalyzer:
    def __init__(self, history_dir: str = "analysis_history", positions_file: str = "open_positions.json"):
        logger.info("Initialisiere MarketAnalyzer")
        self.model = DeepseekAnalyzer()
        self.risk_manager = RiskManager()
        
        # Setze Pfade für Historien- und Positionsdaten
        self.history_dir = Path(history_dir)
        self.positions_file = Path(positions_file)
        
        # Erstelle Verzeichnisse falls nicht vorhanden
        self.history_dir.mkdir(exist_ok=True)
        
        # Initialisiere offene Positionen
        self.open_positions = self._load_positions()
        
        # Portfolio-Tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.current_portfolio_value: float = 0.0

    def _load_positions(self) -> List[Dict[str, Any]]:
        """Lädt bestehende offene Positionen"""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Fehler beim Laden der Positionen: {str(e)}")
            return []

    def _save_positions(self) -> None:
        """Speichert aktuelle offene Positionen"""
        try:
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(self.open_positions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Positionen: {str(e)}")

    def _load_historical_analyses(self, days: int = 7) -> List[Dict[str, Any]]:
        """Lädt historische Analysen der letzten Tage"""
        historical_analyses = []
        current_date = datetime.now()
        
        try:
            for i in range(days):
                date = current_date - timedelta(days=i)
                history_file = self.history_dir / f"analysis_{date.strftime('%Y-%m-%d')}.json"
                
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        historical_analyses.append(json.load(f))
        except Exception as e:
            logger.error(f"Fehler beim Laden der historischen Analysen: {str(e)}")
        
        return historical_analyses

    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Speichert die aktuelle Analyse in der Historie"""
        try:
            current_date = datetime.now()
            history_file = self.history_dir / f"analysis_{current_date.strftime('%Y-%m-%d')}.json"
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Analyse: {str(e)}")

    def update_portfolio_value(self, new_value: float) -> None:
        """Aktualisiert den aktuellen Portfoliowert und speichert ihn in der Historie"""
        self.current_portfolio_value = new_value
        self.portfolio_history.append({
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": new_value
        })
        
        # Analysiere Drawdown
        drawdown_analysis = self.risk_manager.analyze_drawdown(self.portfolio_history)
        if drawdown_analysis["warnungen"]:
            for warnung in drawdown_analysis["warnungen"]:
                logger.warning(f"Drawdown Warnung: {warnung['message']}")

    def validate_new_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert eine neue Position gegen Risikoparameter"""
        if not self.current_portfolio_value:
            logger.error("Kein Portfoliowert gesetzt")
            return {"ist_valid": False, "fehler": "Portfoliowert nicht gesetzt"}
            
        # Berechne optimale Positionsgröße
        sizing_recommendation = self.risk_manager.calculate_position_sizing(
            self.current_portfolio_value,
            self.open_positions,
            position
        )
        
        # Validiere Position
        validation_result = self.risk_manager.validate_position(position, self.current_portfolio_value)
        
        # Füge Sizing-Empfehlung hinzu
        validation_result["position_sizing"] = sizing_recommendation
        
        return validation_result

    def analyze_data(
        self,
        market_data: Optional[str] = None,
        news_data: Optional[str] = None,
        historical_days: int = 7
    ) -> Dict[str, Any]:
        """
        Führt eine umfassende Analyse der bereitgestellten Daten durch und berücksichtigt
        dabei offene Positionen und historische Analysen
        
        Args:
            market_data: Aktuelle Marktdaten
            news_data: Aktuelle Nachrichtendaten
            historical_days: Anzahl der Tage für historische Analysen
            
        Returns:
            Dict mit strukturierten Analyseergebnissen, einschließlich:
            - Handelsempfehlungen
            - Technische Analyse
            - Fundamentale Analyse
            - Risikoeinschätzung
            - Langfristige Strategie
            - Positionsmanagement
            - Historische Trends
        """
        logger.info("Starte umfassende Datenanalyse")
        
        if market_data is None and news_data is None:
            logger.error("Keine Daten für die Analyse bereitgestellt")
            return {
                "fehler": "Keine Analysedaten verfügbar",
                "status": "error"
            }
            
        try:
            # Lade historische Daten und offene Positionen
            historical_analyses = self._load_historical_analyses(historical_days)
            
            # Führe Risikoanalyse durch
            if self.current_portfolio_value:
                risk_assessment = self.risk_manager.calculate_portfolio_risk(
                    self.open_positions,
                    self.current_portfolio_value
                )
            else:
                risk_assessment = {
                    "fehler": "Kein Portfoliowert verfügbar",
                    "warnungen": ["Portfoliowert muss gesetzt sein für vollständige Risikoanalyse"]
                }
            
            # Führe Analyse durch
            analysis_result = self.model.get_combined_analysis(
                market_data=market_data,
                news_data=news_data,
                open_positions=self.open_positions,
                historical_analyses=historical_analyses
            )
            
            # Validiere und bereinige die Ergebnisse
            if not isinstance(analysis_result, dict):
                logger.warning("Analyseergebnis hat unerwartetes Format")
                return {
                    "fehler": "Ungültiges Analyseergebnis-Format",
                    "rohdaten": analysis_result,
                    "status": "error"
                }
            
            # Füge Metadaten und Risikoanalyse hinzu
            analysis_result["analyse_timestamp"] = datetime.now().isoformat()
            analysis_result["analyse_typ"] = "kombiniert" if market_data and news_data else "markt" if market_data else "news"
            analysis_result["anzahl_positionen"] = len(self.open_positions)
            analysis_result["historische_daten_zeitraum"] = f"{historical_days} Tage"
            analysis_result["risiko_analyse"] = risk_assessment
            
            if self.portfolio_history:
                analysis_result["drawdown_analyse"] = self.risk_manager.analyze_drawdown(self.portfolio_history)
            
            # Speichere Analyse in Historie
            self._save_analysis(analysis_result)
            
            # Aktualisiere Positionen basierend auf Empfehlungen
            if "position_management" in analysis_result:
                self._update_positions_from_analysis(analysis_result["position_management"])
            
            # Prüfe und aktualisiere Risikolimits basierend auf Marktbedingungen
            self._update_risk_limits(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            return {
                "fehler": str(e),
                "status": "error"
            }

    def _update_risk_limits(self, analysis_result: Dict[str, Any]) -> None:
        """Aktualisiert Risikolimits basierend auf Marktbedingungen"""
        try:
            market_conditions = analysis_result.get("analyse", {}).get("market_conditions", "normal")
            volatility = analysis_result.get("analyse", {}).get("volatility", "normal")
            
            # Passe Risikolimits an Marktbedingungen an
            new_limits = {}
            
            if market_conditions == "high_risk":
                new_limits.update({
                    "max_position_size": 0.03,  # Reduziere maximale Positionsgröße
                    "stop_loss_minimum": 0.07,  # Erhöhe Stop-Loss Abstände
                    "liquiditaets_reserve": 0.15  # Erhöhe Liquiditätsreserve
                })
            elif market_conditions == "low_risk":
                new_limits.update({
                    "max_position_size": 0.07,  # Erhöhe maximale Positionsgröße
                    "stop_loss_minimum": 0.04  # Reduziere Stop-Loss Abstände
                })
            
            if volatility == "high":
                new_limits.update({
                    "max_leverage": 1.2,  # Reduziere maximalen Hebel
                    "max_position_size": new_limits.get("max_position_size", 0.04)  # Reduziere Positionsgröße weiter
                })
            
            if new_limits:
                self.risk_manager.update_risk_limits(new_limits)
                
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Risikolimits: {str(e)}")

    def _update_positions_from_analysis(self, position_management: Dict[str, Any]) -> None:
        """Aktualisiert Positionen basierend auf den Analyseempfehlungen"""
        try:
            if "bestehende_positionen" in position_management:
                for position_update in position_management["bestehende_positionen"]:
                    position_id = position_update.get("position_id")
                    if position_id:
                        # Validiere Änderungen gegen Risikoparameter
                        position = next((p for p in self.open_positions if p.get("position_id") == position_id), None)
                        if position:
                            updated_position = position.copy()
                            updated_position.update({
                                "letzte_bewertung": position_update.get("aktuelle_bewertung"),
                                "stop_loss": position_update.get("stop_loss_empfehlung"),
                                "take_profit": position_update.get("take_profit_empfehlung"),
                                "anpassung": position_update.get("anpassung_empfehlung"),
                                "letzte_aktualisierung": datetime.now().isoformat()
                            })
                            
                            # Validiere die aktualisierten Werte
                            validation_result = self.validate_new_position(updated_position)
                            if validation_result["ist_valid"]:
                                # Aktualisiere Position
                                for i, pos in enumerate(self.open_positions):
                                    if pos.get("position_id") == position_id:
                                        self.open_positions[i] = updated_position
                                        break
                            else:
                                logger.warning(f"Position {position_id} Update verletzt Risikoparameter: {validation_result['warnungen']}")
            
            self._save_positions()
            
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Positionen: {str(e)}") 