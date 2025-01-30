from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from utils.logging_config import get_logger

class RiskManager:
    """Verwaltet das Risiko für Handelssignale"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Konfigurationswerte extrahieren
        self.max_position_size = config.get("max_position_size", 0.1)
        self.stop_loss_percentage = config.get("stop_loss_percentage", 0.02)
        self.take_profit_percentage = config.get("take_profit_percentage", 0.05)
        self.max_trades_per_day = config.get("max_trades_per_day", 10)
    
    def filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtert und modifiziert Handelssignale basierend auf Risikoparametern"""
        try:
            # Initialisiere gefilterte Signale
            filtered_signals = []
            
            # Gruppiere Signale nach Symbol
            signals_by_symbol = self._group_signals_by_symbol(signals)
            
            # Verarbeite Signale für jedes Symbol
            for symbol, symbol_signals in signals_by_symbol.items():
                # Berechne Risikoscore für jedes Signal
                scored_signals = self._calculate_risk_scores(symbol_signals)
                
                # Wende Risikofilter an
                valid_signals = self._apply_risk_filters(scored_signals)
                
                # Füge gültige Signale hinzu
                filtered_signals.extend(valid_signals)
            
            # Logge Zusammenfassung
            self._log_risk_summary(signals, filtered_signals)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.logger.error(
                f"Fehler beim Filtern der Signale: {str(e)}",
                exc_info=True
            )
            return []
    
    def _group_signals_by_symbol(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Gruppiert Signale nach Symbol"""
        grouped = {}
        for signal in signals:
            symbol = signal.get("symbol")
            if symbol:
                if symbol not in grouped:
                    grouped[symbol] = []
                grouped[symbol].append(signal)
        return grouped
    
    def _calculate_risk_scores(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Berechnet Risikoscores für Signale"""
        scored_signals = []
        
        for signal in signals:
            # Kopiere Signal für Modifikationen
            modified_signal = signal.copy()
            
            # Berechne Basis-Risikoscore
            base_risk_score = self._calculate_base_risk_score(signal)
            
            # Berechne Positions-Risiko
            position_risk = self._calculate_position_risk(signal)
            
            # Berechne Markt-Risiko
            market_risk = self._calculate_market_risk(signal)
            
            # Kombiniere Risikofaktoren
            total_risk_score = (base_risk_score + position_risk + market_risk) / 3
            
            # Füge Risikometriken zum Signal hinzu
            modified_signal.update({
                "risk_score": total_risk_score,
                "base_risk": base_risk_score,
                "position_risk": position_risk,
                "market_risk": market_risk,
                "max_position_size": self._calculate_position_size(total_risk_score)
            })
            
            scored_signals.append(modified_signal)
        
        return scored_signals
    
    def _calculate_base_risk_score(self, signal: Dict[str, Any]) -> float:
        """Berechnet den Basis-Risikoscore"""
        # Verwende Signal-Konfidenz als Basis
        confidence = signal.get("confidence", 0.5)
        
        # Berücksichtige Stop-Loss und Take-Profit
        has_stop_loss = "stop_loss" in signal
        has_take_profit = "take_profit" in signal
        
        # Reduziere Risiko wenn Schutzmaßnahmen vorhanden
        if has_stop_loss and has_take_profit:
            risk_multiplier = 0.8
        elif has_stop_loss:
            risk_multiplier = 0.9
        else:
            risk_multiplier = 1.0
        
        return (1 - confidence) * risk_multiplier
    
    def _calculate_position_risk(self, signal: Dict[str, Any]) -> float:
        """Berechnet das Positionsrisiko"""
        # Beispielhafte Implementierung
        entry_price = signal.get("entry_price", 0)
        stop_loss = signal.get("stop_loss", entry_price * (1 - self.stop_loss_percentage))
        
        # Berechne Risiko basierend auf Stop-Loss-Distanz
        if entry_price > 0:
            risk_percentage = abs(entry_price - stop_loss) / entry_price
            return min(risk_percentage / self.stop_loss_percentage, 1.0)
        return 1.0
    
    def _calculate_market_risk(self, signal: Dict[str, Any]) -> float:
        """Berechnet das Marktrisiko"""
        # Beispielhafte Implementierung
        market_volatility = signal.get("market_volatility", 0.5)
        market_trend = signal.get("market_trend", 0)
        
        # Kombiniere Volatilität und Trend
        return (market_volatility + abs(market_trend)) / 2
    
    def _calculate_position_size(self, risk_score: float) -> float:
        """Berechnet die maximale Positionsgröße basierend auf dem Risikoscore"""
        # Je höher das Risiko, desto kleiner die Position
        return self.max_position_size * (1 - risk_score)
    
    def _apply_risk_filters(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Wendet Risikofilter auf die Signale an"""
        # Sortiere Signale nach Risikoscore (aufsteigend)
        sorted_signals = sorted(signals, key=lambda x: x.get("risk_score", 1.0))
        
        # Filtere Signale basierend auf Risikokriterien
        filtered_signals = []
        daily_trade_count = 0
        
        for signal in sorted_signals:
            # Prüfe maximale Anzahl Trades pro Tag
            if daily_trade_count >= self.max_trades_per_day:
                break
            
            # Prüfe Risikoscore
            if signal.get("risk_score", 1.0) > 1.0:  # Keine Filterung durch Risikoscore
                continue
            
            # Füge Stop-Loss und Take-Profit hinzu, falls nicht vorhanden
            if "stop_loss" not in signal:
                entry_price = signal.get("entry_price", 0)
                signal["stop_loss"] = entry_price * (1 - self.stop_loss_percentage)
            
            if "take_profit" not in signal:
                entry_price = signal.get("entry_price", 0)
                signal["take_profit"] = entry_price * (1 + self.take_profit_percentage)
            
            filtered_signals.append(signal)
            daily_trade_count += 1
        
        return filtered_signals
    
    def _log_risk_summary(self, original_signals: List[Dict[str, Any]], filtered_signals: List[Dict[str, Any]]) -> None:
        """Loggt eine Zusammenfassung der Risikofilterung"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "original_count": len(original_signals),
            "filtered_count": len(filtered_signals),
            "rejection_rate": 1 - (len(filtered_signals) / len(original_signals)) if original_signals else 0,
            "average_risk_score": np.mean([s.get("risk_score", 1.0) for s in filtered_signals]) if filtered_signals else 0
        }
        
        self.logger.log_trade_analysis({
            "risk_summary": summary
        }) 