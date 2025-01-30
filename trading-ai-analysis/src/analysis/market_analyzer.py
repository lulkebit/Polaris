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
from sklearn.preprocessing import StandardScaler

logger = setup_logger(__name__)

@dataclass
class AnalyseConfig:
    lookback_period: int = 30  # Tage für historische Analyse
    min_data_points: int = 20  # Minimum Datenpunkte für technische Analyse
    volatility_window: int = 20  # Fenster für Volatilitätsberechnung
    correlation_threshold: float = 0.6  # Schwellenwert für Korrelationsanalyse (reduziert von 0.7)
    trend_strength_threshold: float = 0.5  # Schwellenwert für Trendstärke (reduziert von 0.6)
    volume_ma_period: int = 20  # Periode für Volumen-Moving-Average

@dataclass
class MarketCondition:
    """Datenklasse für Marktbedingungen"""
    trend: str  # "bullish", "bearish", "neutral"
    volatility: float  # 0-1 Skala
    volume_profile: str  # "high", "normal", "low"
    support_level: float
    resistance_level: float
    market_phase: str  # "accumulation", "distribution", "markup", "markdown"
    risk_level: float  # 0-1 Skala

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

        # Technische Indikatoren konfigurieren
        self.indicators = {
            "sma": {"short": 20, "medium": 50, "long": 200},
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "std_dev": 2},
            "atr": {"period": 14}
        }

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
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Führt eine umfassende Marktanalyse durch"""
        try:
            # Validiere Eingabedaten
            self._validate_market_data(market_data)
            
            # Berechne technische Indikatoren
            technical_indicators = self._calculate_technical_indicators(market_data)
            
            # Analysiere Marktbedingungen
            market_conditions = self._analyze_market_conditions(
                market_data,
                technical_indicators
            )
            
            # Analysiere Volumen und Liquidität
            volume_analysis = self._analyze_volume(market_data)
            
            # Identifiziere wichtige Preisniveaus
            price_levels = self._identify_price_levels(market_data)
            
            # Analysiere Nachrichten und Sentiment (falls verfügbar)
            sentiment_analysis = {}
            if news_data is not None:
                sentiment_analysis = self._analyze_news_sentiment(news_data)
            
            # Kombiniere alle Analyseergebnisse
            analysis_results = {
                "timestamp": datetime.now().isoformat(),
                "market_conditions": market_conditions.__dict__,
                "technical_indicators": technical_indicators,
                "volume_analysis": volume_analysis,
                "price_levels": price_levels,
                "sentiment_analysis": sentiment_analysis
            }
            
            # Logge Analysezusammenfassung
            self._log_analysis_summary(analysis_results)
            
            return analysis_results
        except Exception as e:
            logger.error(f"Fehler bei der Marktanalyse: {str(e)}", exc_info=True)
            raise

    def generate_signals(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert Handelssignale basierend auf der Analyse"""
        try:
            if not analysis_results or not analysis_results:
                logger.warning("Keine Analyseergebnisse verfügbar - Signalgenerierung übersprungen")
                return None
            
            signals = []
            market_conditions = analysis_results.get("market_conditions", {})
            
            # Check if market is tradeable
            if not self._is_market_tradeable(market_conditions):
                return signals
            
            # Generiere Signale basierend auf technischen Indikatoren
            technical_signals = self._generate_technical_signals(
                analysis_results["technical_indicators"],
                market_conditions
            )
            
            # Füge Risikometriken hinzu
            for signal in technical_signals:
                signal.update(self._calculate_signal_metrics(
                    signal,
                    market_conditions,
                    analysis_results
                ))
            
            signals.extend(technical_signals)
            
            # Logge generierte Signale
            self._log_signal_generation(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Fehler bei der Signalgenerierung: {str(e)}", exc_info=True)
            return None

    def _validate_market_data(self, market_data: pd.DataFrame) -> None:
        """Validiert die Eingabedaten"""
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in market_data.columns for col in required_columns):
            raise ValueError(f"Marktdaten müssen folgende Spalten enthalten: {required_columns}")
        
        if market_data.empty:
            raise ValueError("Marktdaten sind leer")
        
        if market_data.isnull().any().any():
            raise ValueError("Marktdaten enthalten NULL-Werte")
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet technische Indikatoren mit der ta Bibliothek"""
        results = {}
        
        try:
            # Trend Indikatoren
            results["sma"] = {
                "short": ta.trend.sma_indicator(data["close"], window=self.indicators["sma"]["short"]),
                "medium": ta.trend.sma_indicator(data["close"], window=self.indicators["sma"]["medium"]),
                "long": ta.trend.sma_indicator(data["close"], window=self.indicators["sma"]["long"])
            }
            
            # RSI
            results["rsi"] = ta.momentum.RSIIndicator(
                data["close"], 
                window=self.indicators["rsi"]["period"]
            ).rsi()
            
            # MACD
            macd = ta.trend.MACD(
                data["close"],
                window_slow=self.indicators["macd"]["slow"],
                window_fast=self.indicators["macd"]["fast"],
                window_sign=self.indicators["macd"]["signal"]
            )
            results["macd"] = {
                "macd": macd.macd(),
                "signal": macd.macd_signal(),
                "hist": macd.macd_diff()
            }
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                data["close"],
                window=self.indicators["bollinger"]["period"],
                window_dev=self.indicators["bollinger"]["std_dev"]
            )
            results["bollinger"] = {
                "high": bollinger.bollinger_hband(),
                "mid": bollinger.bollinger_mavg(),
                "low": bollinger.bollinger_lband()
            }
            
            # ATR
            results["atr"] = ta.volatility.AverageTrueRange(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                window=self.indicators["atr"]["period"]
            ).average_true_range()
            
            # Zusätzliche Indikatoren
            results["volume_vwap"] = ta.volume.VolumeWeightedAveragePrice(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
                window=self.config.volume_ma_period
            ).volume_weighted_average_price()
            
            results["stoch"] = ta.momentum.StochasticOscillator(
                high=data["high"],
                low=data["low"],
                close=data["close"]
            ).stoch()
            
            results["adx"] = ta.trend.ADXIndicator(
                high=data["high"],
                low=data["low"],
                close=data["close"]
            ).adx()
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der technischen Indikatoren: {str(e)}")
            raise
            
        return results
    
    def _analyze_market_conditions(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> MarketCondition:
        """Analysiert die aktuellen Marktbedingungen"""
        # Trend-Analyse
        sma_short = indicators["sma"]["short"].iloc[-1]
        sma_long = indicators["sma"]["long"].iloc[-1]
        
        if sma_short > sma_long * 1.02:
            trend = "bullish"
        elif sma_short < sma_long * 0.98:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Volatilitäts-Analyse
        recent_atr = indicators["atr"][-20:]
        volatility = float(np.mean(recent_atr) / data["close"].iloc[-1])
        
        # Volumen-Analyse
        avg_volume = data["volume"].rolling(window=20).mean()
        current_volume = data["volume"].iloc[-1]
        
        if current_volume > avg_volume.iloc[-1] * 1.5:
            volume_profile = "high"
        elif current_volume < avg_volume.iloc[-1] * 0.5:
            volume_profile = "low"
        else:
            volume_profile = "normal"
        
        # Support und Resistance
        support_level = self._calculate_support(data)
        resistance_level = self._calculate_resistance(data)
        
        # Marktphase bestimmen
        market_phase = self._determine_market_phase(data, indicators)
        
        # Risiko-Level berechnen
        risk_level = self._calculate_risk_level(
            data,
            indicators,
            volatility,
            volume_profile
        )
        
        return MarketCondition(
            trend=trend,
            volatility=volatility,
            volume_profile=volume_profile,
            support_level=support_level,
            resistance_level=resistance_level,
            market_phase=market_phase,
            risk_level=risk_level
        )
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert das Handelsvolumen"""
        return {
            "average_volume": float(data["volume"].mean()),
            "volume_trend": self._calculate_volume_trend(data),
            "volume_spikes": self._detect_volume_spikes(data),
            "volume_profile": self._analyze_volume_profile(data)
        }
    
    def _identify_price_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identifiziert wichtige Preisniveaus"""
        return {
            "support_levels": self._find_support_levels(data),
            "resistance_levels": self._find_resistance_levels(data),
            "pivot_points": self._calculate_pivot_points(data)
        }
    
    def _analyze_news_sentiment(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Nachrichten-Sentiment"""
        if news_data is None or news_data.empty:
            return {}
        
        sentiment_analysis = {
            "overall_sentiment": 0.0,
            "sentiment_by_source": {},
            "key_topics": [],
            "impact_assessment": {}
        }
        
        try:
            # Berechne Gesamt-Sentiment
            if 'sentiment_score' in news_data.columns:
                sentiment_analysis["overall_sentiment"] = float(news_data['sentiment_score'].mean())
            
            # Gruppiere nach Nachrichtenquelle
            if 'source' in news_data.columns and 'sentiment_score' in news_data.columns:
                source_sentiment = news_data.groupby('source')['sentiment_score'].agg(['mean', 'count'])
                sentiment_analysis["sentiment_by_source"] = {
                    source: {
                        "score": float(row['mean']),
                        "count": int(row['count'])
                    }
                    for source, row in source_sentiment.iterrows()
                }
            
            # Identifiziere Hauptthemen
            if 'title' in news_data.columns:
                from collections import Counter
                import re
                
                # Einfache Keyword-Extraktion
                words = ' '.join(news_data['title']).lower()
                words = re.findall(r'\b\w+\b', words)
                common_words = Counter(words).most_common(10)
                sentiment_analysis["key_topics"] = [{"topic": word, "count": count} 
                                                  for word, count in common_words 
                                                  if len(word) > 3]
            
            # Bewerte potenzielle Marktauswirkungen
            if 'impact_score' in news_data.columns:
                sentiment_analysis["impact_assessment"] = {
                    "average_impact": float(news_data['impact_score'].mean()),
                    "high_impact_news": len(news_data[news_data['impact_score'] > 0.7]),
                    "low_impact_news": len(news_data[news_data['impact_score'] < 0.3])
                }
            
            return sentiment_analysis
        
        except Exception as e:
            logger.error(f"Fehler bei der Sentiment-Analyse: {str(e)}")
            return {}
    
    def _is_market_tradeable(self, conditions: Dict[str, Any]) -> bool:
        """Prüft ob der Markt handelbar ist"""
        return (
            conditions["volatility"] < 0.8 and  # Nicht zu volatil
            conditions["risk_level"] < 0.7      # Nicht zu riskant
        )
    
    def _generate_technical_signals(
        self,
        indicators: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generiert Signale basierend auf technischen Indikatoren"""
        signals = []
        
        try:
            for symbol, indicator_data in indicators.items():
                signal = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "signals": [],
                    "strength": 0.0
                }
                
                # RSI-Signale
                if "rsi" in indicator_data:
                    rsi = indicator_data["rsi"].iloc[-1]
                    if rsi < 30:
                        signal["signals"].append({
                            "type": "oversold",
                            "indicator": "RSI",
                            "value": float(rsi),
                            "strength": "strong"
                        })
                    elif rsi > 70:
                        signal["signals"].append({
                            "type": "overbought",
                            "indicator": "RSI",
                            "value": float(rsi),
                            "strength": "strong"
                        })
                
                # MACD-Signale
                if all(k in indicator_data for k in ["macd", "macd_signal"]):
                    macd = indicator_data["macd"].iloc[-1]
                    signal_line = indicator_data["macd_signal"].iloc[-1]
                    
                    if macd > signal_line:
                        signal["signals"].append({
                            "type": "bullish",
                            "indicator": "MACD",
                            "value": float(macd),
                            "strength": "medium"
                        })
                    elif macd < signal_line:
                        signal["signals"].append({
                            "type": "bearish",
                            "indicator": "MACD",
                            "value": float(macd),
                            "strength": "medium"
                        })
                
                # Bollinger Bands Signale
                if all(k in indicator_data for k in ["bb_upper", "bb_lower", "close"]):
                    price = indicator_data["close"].iloc[-1]
                    upper = indicator_data["bb_upper"].iloc[-1]
                    lower = indicator_data["bb_lower"].iloc[-1]
                    
                    if price > upper:
                        signal["signals"].append({
                            "type": "resistance",
                            "indicator": "Bollinger",
                            "value": float(price),
                            "strength": "strong"
                        })
                    elif price < lower:
                        signal["signals"].append({
                            "type": "support",
                            "indicator": "Bollinger",
                            "value": float(price),
                            "strength": "strong"
                        })
                
                # Berechne Gesamtstärke des Signals
                if signal["signals"]:
                    strength_map = {"weak": 0.3, "medium": 0.6, "strong": 1.0}
                    signal["strength"] = sum(strength_map[s["strength"]] for s in signal["signals"]) / len(signal["signals"])
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Fehler bei der Signalgenerierung: {str(e)}")
            return []
    
    def _calculate_signal_metrics(
        self,
        signal: Dict[str, Any],
        conditions: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Berechnet zusätzliche Metriken für ein Signal"""
        return {
            "confidence": self._calculate_signal_confidence(signal, conditions),
            "risk_score": self._calculate_signal_risk(signal, conditions),
            "expected_return": self._calculate_expected_return(signal, analysis),
            "market_context": conditions
        }
    
    def _log_analysis_summary(self, analysis_results: Dict[str, Any]) -> None:
        """Loggt eine Zusammenfassung der Analyseergebnisse"""
        try:
            summary = {
                "timestamp": analysis_results.get("timestamp"),
                "market_conditions": analysis_results.get("market_conditions", {}),
                "num_technical_signals": len(analysis_results.get("technical_indicators", {})),
                "volume_analysis": analysis_results.get("volume_analysis", {}),
                "price_levels": analysis_results.get("price_levels", {}),
                "sentiment_analysis": analysis_results.get("sentiment_analysis", {})
            }
            
            # Verwende die Standard-Logging-Methode
            logger.info("Analysezusammenfassung", extra={"summary": summary})
            
        except Exception as e:
            logger.error(f"Fehler beim Loggen der Analysezusammenfassung: {str(e)}")
    
    def _log_signal_generation(self, signals: List[Dict[str, Any]]) -> None:
        """Loggt generierte Signale"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_signals": len(signals),
            "signal_types": self._count_signal_types(signals),
            "average_confidence": np.nanmean([s.get("confidence", 0) for s in signals]) if signals else 0
        }
        
        logger.log_trade_analysis({"signal_generation": summary})

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
                    "volume_trend": self._calculate_volume_trend(market_df)
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

    def _calculate_support(self, data: pd.DataFrame, window: int = 20) -> float:
        """Berechnet das Support-Level basierend auf lokalen Minimums"""
        recent_data = data.tail(window)
        return float(recent_data['low'].min())

    def _calculate_resistance(self, data: pd.DataFrame, window: int = 20) -> float:
        """Berechnet das Resistance-Level basierend auf lokalen Maximums"""
        recent_data = data.tail(window)
        return float(recent_data['high'].max())

    def _calculate_risk_level(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Any],
        volatility: float,
        volume_profile: str
    ) -> float:
        """Berechnet das Risiko-Level basierend auf verschiedenen Faktoren"""
        risk_score = 0.0
        
        # Volatilitäts-basiertes Risiko (0-0.4)
        if volatility > 0.03:  # Hohe Volatilität
            risk_score += 0.4
        elif volatility > 0.02:  # Mittlere Volatilität
            risk_score += 0.2
        else:  # Niedrige Volatilität
            risk_score += 0.1
            
        # Volumen-basiertes Risiko (0-0.3)
        if volume_profile == "high":
            risk_score += 0.3
        elif volume_profile == "normal":
            risk_score += 0.15
            
        # Trend-basiertes Risiko (0-0.3)
        rsi = indicators.get("rsi", pd.Series([50])).iloc[-1]
        if rsi > 70 or rsi < 30:  # Überkauft oder überverkauft
            risk_score += 0.3
        elif 40 <= rsi <= 60:  # Neutraler Bereich
            risk_score += 0.15
            
        return min(1.0, risk_score)

    def _determine_market_phase(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Bestimmt die aktuelle Marktphase"""
        # Trend-Analyse
        sma_short = indicators["sma"]["short"].iloc[-1]
        sma_medium = indicators["sma"]["medium"].iloc[-1]
        sma_long = indicators["sma"]["long"].iloc[-1]
        current_price = data["close"].iloc[-1]
        
        if current_price > sma_short > sma_medium > sma_long:
            return "markup"  # Starker Aufwärtstrend
        elif current_price < sma_short < sma_medium < sma_long:
            return "markdown"  # Starker Abwärtstrend
        elif sma_short < current_price < sma_medium and sma_long < sma_medium:
            return "accumulation"  # Mögliche Bodenbildung
        elif sma_short > current_price > sma_medium and sma_long > sma_medium:
            return "distribution"  # Mögliche Topbildung
        else:
            return "sideways"  # Seitwärtsbewegung

    def _calculate_volume_trend(self, data: pd.DataFrame, window: int = 20) -> str:
        """Berechnet den Volumentrend basierend auf gleitendem Durchschnitt"""
        if 'volume' not in data.columns:
            return "unknown"
            
        volume = data['volume']
        volume_ma = volume.rolling(window=window).mean()
        
        # Vergleiche aktuelles Volumen mit gleitendem Durchschnitt
        current_volume_ma = volume_ma.iloc[-1]
        past_volume_ma = volume_ma.iloc[-window]
        
        if current_volume_ma > past_volume_ma * 1.1:
            return "increasing"
        elif current_volume_ma < past_volume_ma * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _detect_volume_spikes(self, data: pd.DataFrame, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Erkennt signifikante Volumenspitzen"""
        if 'volume' not in data.columns:
            return []
            
        volume = data['volume']
        volume_ma = volume.rolling(window=20).mean()
        volume_std = volume.rolling(window=20).std()
        
        spikes = []
        for i in range(len(data)):
            if volume.iloc[i] > volume_ma.iloc[i] + (threshold * volume_std.iloc[i]):
                spikes.append({
                    "timestamp": data.index[i],
                    "volume": float(volume.iloc[i]),
                    "average_volume": float(volume_ma.iloc[i]),
                    "deviation": float((volume.iloc[i] - volume_ma.iloc[i]) / volume_std.iloc[i])
                })
        
        return spikes

    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert das Volumprofil für verschiedene Preisbereiche"""
        if 'volume' not in data.columns:
            return {}
            
        # Teile den Preisbereich in Zonen auf
        price_range = data['high'].max() - data['low'].min()
        zone_size = price_range / 10
        
        volume_profile = {}
        for i in range(10):
            zone_low = data['low'].min() + (i * zone_size)
            zone_high = zone_low + zone_size
            
            # Finde Volumen in dieser Preiszone
            zone_mask = (data['low'] >= zone_low) & (data['high'] < zone_high)
            zone_volume = data.loc[zone_mask, 'volume'].sum()
            
            volume_profile[f"zone_{i+1}"] = {
                "price_range": [float(zone_low), float(zone_high)],
                "volume": float(zone_volume)
            }
        
        return {
            "volume_by_price_zone": volume_profile,
            "total_volume": float(data['volume'].sum()),
            "average_daily_volume": float(data['volume'].mean())
        }

    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """
        Identifiziert Unterstützungsniveaus basierend auf den historischen Preisdaten.
        
        Args:
            data: DataFrame mit historischen Preisdaten
            
        Returns:
            Liste von Unterstützungsniveaus
        """
        try:
            # Berechne gleitende Durchschnitte für Unterstützungsniveaus
            short_ma = data['low'].rolling(window=5).mean()
            long_ma = data['low'].rolling(window=20).mean()
            
            # Finde Kreuzungspunkte, die auf Unterstützung hindeuten
            crossovers = np.where(short_ma > long_ma, 1, 0)
            support_levels = []
            
            for i in range(1, len(crossovers)):
                if crossovers[i] == 1 and crossovers[i-1] == 0:
                    support_levels.append(data['low'].iloc[i])
            
            # Entferne Duplikate und sortiere die Niveaus
            support_levels = sorted(list(set(support_levels)))
            
            return support_levels[-3:]  # Rückgabe der letzten 3 Unterstützungsniveaus
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Unterstützungsniveaus: {str(e)}")
            return []

    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """
        Identifiziert Widerstandsniveaus basierend auf den historischen Preisdaten.
        
        Args:
            data: DataFrame mit historischen Preisdaten
            
        Returns:
            Liste von Widerstandsniveaus
        """
        try:
            # Berechne gleitende Durchschnitte für Widerstandsniveaus
            short_ma = data['high'].rolling(window=5).mean()
            long_ma = data['high'].rolling(window=20).mean()
            
            # Finde Kreuzungspunkte, die auf Widerstand hindeuten
            crossovers = np.where(short_ma < long_ma, 1, 0)
            resistance_levels = []
            
            for i in range(1, len(crossovers)):
                if crossovers[i] == 1 and crossovers[i-1] == 0:
                    resistance_levels.append(data['high'].iloc[i])
            
            # Entferne Duplikate und sortiere die Niveaus
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return resistance_levels[-3:]  # Rückgabe der letzten 3 Widerstandsniveaus
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Widerstandsniveaus: {str(e)}")
            return []

    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet Pivot-Punkte basierend auf historischen Preisdaten"""
        try:
            if len(data) < 1:
                return {}
            
            # Hole letzte Handelssession
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            # Berechne klassische Pivot-Punkte
            pivot = (high + low + close) / 3
            
            # Support Levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # Resistance Levels
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # Fibonacci Pivot-Punkte
            fib_r3 = pivot + ((high - low) * 1.618)
            fib_r2 = pivot + ((high - low) * 1.272)
            fib_r1 = pivot + ((high - low) * 0.618)
            fib_s1 = pivot - ((high - low) * 0.618)
            fib_s2 = pivot - ((high - low) * 1.272)
            fib_s3 = pivot - ((high - low) * 1.618)
            
            return {
                "classic": {
                    "pivot": float(pivot),
                    "support": {
                        "s1": float(s1),
                        "s2": float(s2),
                        "s3": float(s3)
                    },
                    "resistance": {
                        "r1": float(r1),
                        "r2": float(r2),
                        "r3": float(r3)
                    }
                },
                "fibonacci": {
                    "pivot": float(pivot),
                    "support": {
                        "s1": float(fib_s1),
                        "s2": float(fib_s2),
                        "s3": float(fib_s3)
                    },
                    "resistance": {
                        "r1": float(fib_r1),
                        "r2": float(fib_r2),
                        "r3": float(fib_r3)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Pivot-Punkte: {str(e)}")
            return {}

    def _calculate_signal_confidence(self, signal: Dict[str, Any], conditions: Dict[str, Any]) -> float:
        """Berechnet die Vertrauenswürdigkeit eines Signals"""
        # Kombiniere technische Indikatoren und Marktbedingungen
        confidence = 0.5
        
        # Erhöhe Vertrauen bei starken Trends
        if conditions.get("trend_strength", 0) > 0.7:
            confidence += 0.2
            
        # Erhöhe Vertrauen bei hohem Volumen
        if conditions.get("volume_profile") == "high":
            confidence += 0.1
            
        # Begrenze auf 0-1 Bereich
        return min(max(confidence, 0), 1)

    def _calculate_signal_risk(self, signal: Dict[str, Any], conditions: Dict[str, Any]) -> float:
        """Berechnet das Risiko eines Signals"""
        try:
            base_risk = 0.5  # Basis-Risiko
            
            # Marktbedingungen einbeziehen
            if conditions.get("volatility", 0) > 0.2:
                base_risk += 0.2
            if conditions.get("market_regime") in ["volatile_bearish", "highly_volatile"]:
                base_risk += 0.15
            
            # Signal-spezifische Faktoren
            if signal.get("type") in ["oversold", "overbought"]:
                base_risk += 0.1
            if signal.get("strength", 0) < 0.5:
                base_risk += 0.1
            
            # Volumen-basierte Anpassung
            if conditions.get("volume_profile") == "low":
                base_risk += 0.1
            
            return min(max(base_risk, 0), 1)  # Begrenze auf 0-1
            
        except Exception as e:
            logger.error(f"Fehler bei der Risikoberechnung: {str(e)}")
            return 0.5

    def _calculate_expected_return(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Berechnet den erwarteten Rückfluss eines Signals"""
        try:
            base_return = 0.0
            
            # Signal-Stärke einbeziehen
            signal_strength = signal.get("strength", 0.5)
            base_return += signal_strength * 0.05
            
            # Trend-Alignment
            if analysis.get("trend_direction") == signal.get("type"):
                base_return += 0.02
            
            # Marktbedingungen
            market_conditions = analysis.get("market_conditions", {})
            if market_conditions.get("regime") in ["bullish", "low_vol_bullish"]:
                base_return += 0.02
            
            # Volumen-Profil
            if analysis.get("volume_analysis", {}).get("volume_trend") == "increasing":
                base_return += 0.01
            
            # Support/Resistance Nähe
            price_levels = analysis.get("price_levels", {})
            if price_levels and signal.get("price"):
                nearest_support = min((abs(level - signal["price"]) 
                                     for level in price_levels.get("support_levels", [])), 
                                    default=float('inf'))
                if nearest_support < 0.02:  # Wenn Preis nahe Support
                    base_return += 0.02
                
            return max(base_return, 0)  # Keine negativen Erwartungen
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung des erwarteten Returns: {str(e)}")
            return 0.0

    def _count_signal_types(self, signals: List[Dict[str, Any]]) -> Dict[str, int]:
        """Zählt die verschiedenen Signaltypen"""
        signal_types = {}
        for signal in signals:
            signal_type = signal.get("type", "unknown")
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        return signal_types

    def analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Marktbedingungen und generiert Signale"""
        try:
            # Calculate indicators first
            indicators = self._calculate_technical_indicators(market_data)
            
            # Then analyze market conditions with the indicators
            conditions = self._analyze_market_conditions(market_data, indicators)
            
            # Detailliertes Logging der Marktbedingungen
            logger.info(
                "Marktbedingungen analysiert",
                extra={
                    "market_conditions": {
                        "trend": conditions.trend,
                        "volatility": conditions.volatility,
                        "volume_profile": conditions.volume_profile,
                        "market_phase": conditions.market_phase,
                        "risk_level": conditions.risk_level
                    }
                }
            )
            
            return {
                "market_conditions": conditions.__dict__,
                "technical_indicators": indicators
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Marktanalyse: {str(e)}", exc_info=True)
            return {} 