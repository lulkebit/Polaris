from models.deepseek_model import DeepseekAnalyzer
from utils.logger import setup_logger
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from risk.risk_manager import RiskManager
import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass

logger = setup_logger(__name__)

@dataclass
class AnalyseConfig:
    lookback_period: int = 30  # Tage für historische Analyse
    min_data_points: int = 20  # Minimum Datenpunkte für technische Analyse
    volatility_window: int = 20  # Fenster für Volatilitätsberechnung
    correlation_threshold: float = 0.7  # Schwellenwert für Korrelationsanalyse
    trend_strength_threshold: float = 0.6  # Schwellenwert für Trendstärke
    volume_ma_period: int = 20  # Periode für Volumen-Moving-Average

class MarketAnalyzer:
    def __init__(self, history_dir: str = "analysis_history", positions_file: str = "open_positions.json"):
        logger.info("Initialisiere MarketAnalyzer")
        self.model = DeepseekAnalyzer()
        self.risk_manager = RiskManager()
        self.config = AnalyseConfig()
        
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
        market_data: Optional[pd.DataFrame] = None,
        news_data: Optional[pd.DataFrame] = None,
        historical_days: int = 7
    ) -> Dict[str, Any]:
        """
        Führt eine umfassende Analyse der bereitgestellten Daten durch und berücksichtigt
        dabei offene Positionen und historische Analysen
        
        Args:
            market_data: Aktuelle Marktdaten als DataFrame
            news_data: Aktuelle Nachrichtendaten als DataFrame
            historical_days: Anzahl der Tage für historische Analysen
            
        Returns:
            Dict mit strukturierten Analyseergebnissen, einschließlich:
            - Handelsempfehlungen
            - Technische Analyse
            - Fundamentale Analyse
            - Risikoeinschätzung
            - Langfristige Strategie
        """
        logger.info("Starte umfassende Datenanalyse")
        
        if market_data is None and news_data is None:
            logger.error("Keine Daten für die Analyse bereitgestellt")
            return {}
            
        try:
            # Konvertiere DataFrames in String-Repräsentation für das Modell
            market_data_str = market_data.to_json(orient='records', date_format='iso') if market_data is not None else ""
            news_data_str = news_data.to_json(orient='records', date_format='iso') if news_data is not None else ""
            
            # Lade historische Analysen
            historical_analyses = self._load_historical_analyses(days=historical_days)
            
            # Führe KI-Analyse durch
            analysis_results = self.model.get_combined_analysis(
                market_data=market_data_str,
                news_data=news_data_str,
                open_positions=self.open_positions,
                historical_analyses=historical_analyses
            )
            
            # Speichere Analyse in der Historie
            self._save_analysis(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            return {}

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

    def _analyze_market_data(self, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Marktdaten"""
        analysis = {
            "volatility": {},
            "volume_analysis": {},
            "price_levels": {},
            "market_regime": "unknown"
        }
        
        for symbol in market_df.columns:
            if symbol == 'timestamp':
                continue
                
            prices = market_df[symbol].dropna()
            if len(prices) < self.config.min_data_points:
                continue
            
            # Volatilitätsanalyse
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualisierte Volatilität
            
            analysis["volatility"][symbol] = {
                "annual_volatility": volatility,
                "recent_volatility": returns.tail(5).std() * np.sqrt(252)
            }
            
            # Volumenanalyse (falls verfügbar)
            if f"{symbol}_volume" in market_df.columns:
                volume = market_df[f"{symbol}_volume"]
                volume_ma = volume.rolling(window=self.config.volume_ma_period).mean()
                
                analysis["volume_analysis"][symbol] = {
                    "avg_volume": volume.mean(),
                    "volume_trend": "increasing" if volume.tail(5).mean() > volume_ma.tail(5).mean() else "decreasing"
                }
            
            # Wichtige Preisniveaus
            analysis["price_levels"][symbol] = {
                "current_price": prices.iloc[-1],
                "sma_20": prices.rolling(window=20).mean().iloc[-1],
                "sma_50": prices.rolling(window=50).mean().iloc[-1],
                "recent_high": prices.tail(20).max(),
                "recent_low": prices.tail(20).min()
            }
        
        # Marktregime-Bestimmung
        analysis["market_regime"] = self._determine_market_regime(market_df)
        
        return analysis

    def _analyze_news_data(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Nachrichtendaten"""
        if news_df is None or news_df.empty:
            return {}
            
        news_analysis = {
            "sentiment_summary": {},
            "key_events": [],
            "sector_impact": {}
        }
        
        # Sentiment-Analyse pro Symbol/Sektor
        if 'symbol' in news_df.columns and 'sentiment' in news_df.columns:
            sentiment_grouped = news_df.groupby('symbol')['sentiment'].agg(['mean', 'count'])
            
            for symbol, data in sentiment_grouped.iterrows():
                news_analysis["sentiment_summary"][symbol] = {
                    "avg_sentiment": data['mean'],
                    "news_count": data['count']
                }
        
        # Identifiziere wichtige Events
        if 'importance' in news_df.columns:
            important_news = news_df[news_df['importance'] > 0.7]
            news_analysis["key_events"] = important_news.to_dict('records')
        
        return news_analysis

    def _calculate_technical_indicators(self, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet technische Indikatoren"""
        indicators = {}
        
        for symbol in market_df.columns:
            if symbol == 'timestamp':
                continue
                
            close_prices = market_df[symbol].dropna()
            if len(close_prices) < self.config.min_data_points:
                continue
            
            # Trend Indikatoren
            indicators[symbol] = {
                "rsi": ta.momentum.RSIIndicator(close_prices).rsi(),
                "macd": ta.trend.MACD(close_prices).macd(),
                "macd_signal": ta.trend.MACD(close_prices).macd_signal(),
                "macd_diff": ta.trend.MACD(close_prices).macd_diff(),
                "sma_20": ta.trend.SMAIndicator(close_prices, window=20).sma_indicator(),
                "sma_50": ta.trend.SMAIndicator(close_prices, window=50).sma_indicator(),
                "ema_20": ta.trend.EMAIndicator(close_prices, window=20).ema_indicator(),
                
                # Volatilitäts Indikatoren
                "bollinger_high": ta.volatility.BollingerBands(close_prices).bollinger_hband(),
                "bollinger_low": ta.volatility.BollingerBands(close_prices).bollinger_lband(),
                "atr": ta.volatility.AverageTrueRange(
                    high=market_df[f"{symbol}_high"] if f"{symbol}_high" in market_df else close_prices,
                    low=market_df[f"{symbol}_low"] if f"{symbol}_low" in market_df else close_prices,
                    close=close_prices
                ).average_true_range(),
                
                # Volumen Indikatoren
                "obv": ta.volume.OnBalanceVolumeIndicator(
                    close_prices,
                    market_df[f"{symbol}_volume"] if f"{symbol}_volume" in market_df else pd.Series([0] * len(close_prices))
                ).on_balance_volume()
            }
        
        return indicators

    def _analyze_trends(self, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Markttrends"""
        trends = {}
        
        for symbol in market_df.columns:
            if symbol == 'timestamp':
                continue
                
            prices = market_df[symbol].dropna()
            if len(prices) < self.config.min_data_points:
                continue
            
            # Trendstärke berechnen
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            
            current_price = prices.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            # Trendrichtung bestimmen
            trend_direction = "seitwärts"
            if current_price > current_sma_20 > current_sma_50:
                trend_direction = "aufwärts"
            elif current_price < current_sma_20 < current_sma_50:
                trend_direction = "abwärts"
            
            # Trendstärke berechnen
            returns = prices.pct_change()
            trend_strength = abs(returns.mean()) / returns.std() if returns.std() != 0 else 0
            
            trends[symbol] = {
                "direction": trend_direction,
                "strength": trend_strength,
                "above_sma_20": current_price > current_sma_20,
                "above_sma_50": current_price > current_sma_50,
                "sma_alignment": "bullish" if current_sma_20 > current_sma_50 else "bearish"
            }
        
        return trends

    def _generate_trading_signals(
        self,
        market_df: pd.DataFrame,
        technical_indicators: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generiert Handelssignale"""
        signals = []
        
        for symbol in market_df.columns:
            if symbol == 'timestamp':
                continue
                
            if symbol not in technical_indicators or symbol not in trend_analysis:
                continue
            
            indicators = technical_indicators[symbol]
            trend = trend_analysis[symbol]
            
            # RSI Signale
            rsi = indicators["rsi"].iloc[-1] if not pd.isna(indicators["rsi"].iloc[-1]) else 50
            
            # MACD Signale
            macd = indicators["macd"].iloc[-1]
            macd_signal = indicators["macd_signal"].iloc[-1]
            macd_diff = indicators["macd_diff"].iloc[-1]
            
            # Bollinger Bands
            bb_high = indicators["bollinger_high"].iloc[-1]
            bb_low = indicators["bollinger_low"].iloc[-1]
            current_price = market_df[symbol].iloc[-1]
            
            signal = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "signals": []
            }
            
            # RSI Überverkauft/Überkauft
            if rsi < 30:
                signal["signals"].append({
                    "type": "oversold",
                    "indicator": "RSI",
                    "value": rsi,
                    "strength": "strong" if rsi < 20 else "moderate"
                })
            elif rsi > 70:
                signal["signals"].append({
                    "type": "overbought",
                    "indicator": "RSI",
                    "value": rsi,
                    "strength": "strong" if rsi > 80 else "moderate"
                })
            
            # MACD Kreuzungen
            if macd_diff > 0 and abs(macd - macd_signal) < 0.1:
                signal["signals"].append({
                    "type": "bullish",
                    "indicator": "MACD",
                    "value": macd_diff,
                    "strength": "strong" if macd_diff > 0.2 else "moderate"
                })
            elif macd_diff < 0 and abs(macd - macd_signal) < 0.1:
                signal["signals"].append({
                    "type": "bearish",
                    "indicator": "MACD",
                    "value": macd_diff,
                    "strength": "strong" if macd_diff < -0.2 else "moderate"
                })
            
            # Bollinger Band Signale
            if current_price <= bb_low:
                signal["signals"].append({
                    "type": "support",
                    "indicator": "Bollinger",
                    "value": current_price,
                    "strength": "strong"
                })
            elif current_price >= bb_high:
                signal["signals"].append({
                    "type": "resistance",
                    "indicator": "Bollinger",
                    "value": current_price,
                    "strength": "strong"
                })
            
            if signal["signals"]:
                signals.append(signal)
        
        return signals

    def _determine_market_regime(self, market_df: pd.DataFrame) -> str:
        """Bestimmt das aktuelle Marktregime"""
        # Berechne durchschnittliche Volatilität und Trend
        returns = market_df.select_dtypes(include=[np.number]).pct_change()
        volatility = returns.std().mean() * np.sqrt(252)
        trend = returns.mean().mean() * 252
        
        if volatility > 0.25:  # Hohe Volatilität
            if trend > 0.05:
                return "volatile_bullish"
            elif trend < -0.05:
                return "volatile_bearish"
            return "highly_volatile"
        elif volatility < 0.10:  # Niedrige Volatilität
            if trend > 0.02:
                return "low_vol_bullish"
            elif trend < -0.02:
                return "low_vol_bearish"
            return "low_vol_sideways"
        else:  # Moderate Volatilität
            if trend > 0.03:
                return "bullish"
            elif trend < -0.03:
                return "bearish"
            return "sideways"

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert Handelsempfehlungen basierend auf der Analyse"""
        recommendations = []
        
        # Kombiniere verschiedene Analyseergebnisse
        for symbol in analysis_result.get("technical_indicators", {}):
            recommendation = {
                "symbol": symbol,
                "action": None,
                "confidence": 0.0,
                "reasons": [],
                "risk_level": "medium",
                "suggested_size": 0.0
            }
            
            # Technische Signale auswerten
            signals = [s for s in analysis_result["trading_signals"] if s["symbol"] == symbol]
            if signals:
                signal_strength = sum(1 for s in signals[0]["signals"] if s["strength"] == "strong")
                recommendation["confidence"] = min(0.8, signal_strength * 0.2)
                
                # Bestimme Handelsrichtung
                bullish_signals = sum(1 for s in signals[0]["signals"] if s["type"] in ["bullish", "oversold", "support"])
                bearish_signals = sum(1 for s in signals[0]["signals"] if s["type"] in ["bearish", "overbought", "resistance"])
                
                if bullish_signals > bearish_signals:
                    recommendation["action"] = "buy"
                    recommendation["reasons"].append("Überwiegend bullische technische Signale")
                elif bearish_signals > bullish_signals:
                    recommendation["action"] = "sell"
                    recommendation["reasons"].append("Überwiegend bärische technische Signale")
            
            # Trend-Analyse einbeziehen
            if symbol in analysis_result.get("trend_analysis", {}):
                trend = analysis_result["trend_analysis"][symbol]
                if trend["strength"] > self.config.trend_strength_threshold:
                    if trend["direction"] == "aufwärts" and recommendation["action"] != "sell":
                        recommendation["confidence"] += 0.1
                        recommendation["reasons"].append("Starker Aufwärtstrend")
                    elif trend["direction"] == "abwärts" and recommendation["action"] != "buy":
                        recommendation["confidence"] += 0.1
                        recommendation["reasons"].append("Starker Abwärtstrend")
            
            # Risikobewertung
            if "risk_assessment" in analysis_result:
                risk_metrics = analysis_result["risk_assessment"]
                if symbol in risk_metrics.get("position_sizes", {}):
                    current_size = risk_metrics["position_sizes"][symbol]
                    if current_size > 0.15:  # Große Position
                        recommendation["risk_level"] = "high"
                    elif current_size < 0.05:  # Kleine Position
                        recommendation["risk_level"] = "low"
            
            # Position Sizing
            if recommendation["action"] and recommendation["confidence"] > 0.5:
                base_size = 0.05  # 5% Basis-Positionsgröße
                confidence_adjustment = (recommendation["confidence"] - 0.5) * 0.1
                risk_adjustment = -0.02 if recommendation["risk_level"] == "high" else 0.02 if recommendation["risk_level"] == "low" else 0
                
                recommendation["suggested_size"] = min(0.15, base_size + confidence_adjustment + risk_adjustment)
            
            if recommendation["action"]:
                recommendations.append(recommendation)
        
        return recommendations

    def _analyze_portfolio(self) -> Dict[str, Any]:
        """Analysiert das aktuelle Portfolio"""
        return {
            "current_value": self.current_portfolio_value,
            "open_positions": len(self.open_positions),
            "position_distribution": self._calculate_position_distribution(),
            "recent_performance": self._calculate_recent_performance()
        }

    def _calculate_position_distribution(self) -> Dict[str, float]:
        """Berechnet die Verteilung der Positionen"""
        if not self.current_portfolio_value or not self.open_positions:
            return {}
            
        return {
            symbol: pos['value'] / self.current_portfolio_value
            for symbol, pos in self.open_positions.items()
        }

    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Berechnet die jüngste Performance"""
        if len(self.portfolio_history) < 2:
            return {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0
            }
            
        current_value = self.portfolio_history[-1]["portfolio_value"]
        
        # Tagesrendite
        one_day_ago = datetime.now() - timedelta(days=1)
        daily_start_value = next(
            (h["portfolio_value"] for h in reversed(self.portfolio_history) if h["timestamp"] < one_day_ago),
            self.portfolio_history[0]["portfolio_value"]
        )
        
        # Wochenrendite
        one_week_ago = datetime.now() - timedelta(days=7)
        weekly_start_value = next(
            (h["portfolio_value"] for h in reversed(self.portfolio_history) if h["timestamp"] < one_week_ago),
            self.portfolio_history[0]["portfolio_value"]
        )
        
        # Monatsrendite
        one_month_ago = datetime.now() - timedelta(days=30)
        monthly_start_value = next(
            (h["portfolio_value"] for h in reversed(self.portfolio_history) if h["timestamp"] < one_month_ago),
            self.portfolio_history[0]["portfolio_value"]
        )
        
        return {
            "daily_return": (current_value - daily_start_value) / daily_start_value,
            "weekly_return": (current_value - weekly_start_value) / weekly_start_value,
            "monthly_return": (current_value - monthly_start_value) / monthly_start_value
        } 