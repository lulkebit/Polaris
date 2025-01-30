from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from ..backtesting.backtester import Backtester, BacktestConfig, BacktestResults
from ..models.deepseek_model import DeepseekAnalyzer
from ..risk.risk_manager import RiskManager
from ..utils.logger import setup_logger
import json

logger = setup_logger(__name__)

class StrategyOptimizer:
    def __init__(self):
        self.model = DeepseekAnalyzer()
        self.risk_manager = RiskManager()
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_strategy: Optional[Dict[str, Any]] = None
        self.best_results: Optional[BacktestResults] = None
        
        # Optimierungsziele
        self.target_metrics = {
            'min_sharpe_ratio': 1.5,
            'max_drawdown_limit': 0.15,
            'min_win_rate': 0.55,
            'min_profit_factor': 1.5,
            'min_annual_return': 0.12,
            'max_volatility': 0.20
        }

    def optimize_strategy(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
        max_iterations: int = 10,
        optimization_period: str = "1Y"
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """
        Führt eine KI-gesteuerte Strategieoptimierung durch
        """
        logger.info("Starte KI-gesteuerte Strategieoptimierung")
        
        try:
            for iteration in range(max_iterations):
                logger.info(f"Optimierungsiteration {iteration + 1}/{max_iterations}")
                
                # Analysiere bisherige Ergebnisse und generiere neue Strategie
                strategy_config = self._generate_strategy_config(iteration)
                
                # Führe Backtest durch
                results = self._run_backtest(market_data, news_data, strategy_config)
                
                # Analysiere Ergebnisse
                analysis = self._analyze_results(results, strategy_config)
                
                # Speichere Ergebnisse
                self.optimization_history.append({
                    'iteration': iteration,
                    'strategy_config': strategy_config,
                    'results': results,
                    'analysis': analysis
                })
                
                # Aktualisiere beste Strategie
                if self._is_better_strategy(results):
                    self.best_strategy = strategy_config
                    self.best_results = results
                    logger.info("Neue beste Strategie gefunden!")
                
                # Prüfe, ob Ziele erreicht wurden
                if self._goals_achieved(results):
                    logger.info("Optimierungsziele erreicht!")
                    break
                
                # Lasse KI die Ergebnisse analysieren und Verbesserungen vorschlagen
                self._ai_analyze_and_improve(iteration)
            
            if self.best_strategy is None:
                logger.warning("Keine zufriedenstellende Strategie gefunden")
                self.best_strategy = self.optimization_history[-1]['strategy_config']
                self.best_results = self.optimization_history[-1]['results']
            
            return self.best_strategy, self.best_results
            
        except Exception as e:
            logger.error(f"Fehler während der Strategieoptimierung: {str(e)}")
            raise

    def _generate_strategy_config(self, iteration: int) -> Dict[str, Any]:
        """Generiert eine neue Strategiekonfiguration basierend auf bisherigen Ergebnissen"""
        try:
            # Erstelle Prompt für die KI
            history_data = json.dumps(self.optimization_history) if self.optimization_history else "Keine bisherigen Daten"
            
            prompt = f"""Analysiere die bisherigen Optimierungsergebnisse und generiere eine verbesserte Handelsstrategie-Konfiguration.

            Optimierungshistorie:
            {history_data}

            Aktuelle Iteration: {iteration}
            
            Optimierungsziele:
            {json.dumps(self.target_metrics, indent=2)}

            Erstelle eine Konfiguration im folgenden Format:
            {{
                "risk_management": {{
                    "position_size_limit": float,
                    "stop_loss_atr_multiple": float,
                    "take_profit_atr_multiple": float,
                    "max_positions": int,
                    "sector_exposure_limit": float
                }},
                "entry_conditions": {{
                    "min_volatility": float,
                    "min_volume": float,
                    "trend_strength": float,
                    "correlation_threshold": float
                }},
                "exit_conditions": {{
                    "profit_taking_threshold": float,
                    "max_loss_threshold": float,
                    "trend_reversal_threshold": float
                }},
                "timing": {{
                    "holding_period": str,
                    "rebalancing_frequency": str,
                    "entry_timing_factors": List[str]
                }}
            }}
            """
            
            # Hole KI-Empfehlung
            strategy_config = self.model.generate_strategy_config(prompt)
            
            # Validiere und bereinige Konfiguration
            strategy_config = self._validate_strategy_config(strategy_config)
            
            return strategy_config
            
        except Exception as e:
            logger.error(f"Fehler bei der Strategiegenerierung: {str(e)}")
            # Fallback auf Standardkonfiguration
            return self._get_default_strategy_config()

    def _run_backtest(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame],
        strategy_config: Dict[str, Any]
    ) -> BacktestResults:
        """Führt einen Backtest mit der gegebenen Konfiguration durch"""
        
        # Erstelle Backtester-Konfiguration
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage=0.001,
            max_positions=strategy_config['risk_management']['max_positions'],
            rebalancing_frequency=strategy_config['timing']['rebalancing_frequency']
        )
        
        # Initialisiere Backtester
        backtester = Backtester(config)
        
        # Führe Backtest durch
        results = backtester.run_backtest(market_data, news_data)
        
        return results

    def _analyze_results(self, results: BacktestResults, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert die Backtest-Ergebnisse"""
        analysis = {
            'metrics_evaluation': {
                'sharpe_ratio': {
                    'value': results.sharpe_ratio,
                    'target_achieved': results.sharpe_ratio >= self.target_metrics['min_sharpe_ratio']
                },
                'max_drawdown': {
                    'value': results.max_drawdown,
                    'target_achieved': results.max_drawdown <= self.target_metrics['max_drawdown_limit']
                },
                'win_rate': {
                    'value': results.win_rate,
                    'target_achieved': results.win_rate >= self.target_metrics['min_win_rate']
                },
                'profit_factor': {
                    'value': results.profit_factor,
                    'target_achieved': results.profit_factor >= self.target_metrics['min_profit_factor']
                }
            },
            'risk_assessment': {
                'volatility': results.risk_metrics['volatility'],
                'var_95': results.risk_metrics['var_95'],
                'beta': results.risk_metrics['beta']
            },
            'strategy_effectiveness': {
                'avg_trade_duration': results.performance_summary['avg_trade_duration'],
                'avg_profit_per_trade': results.performance_summary['avg_profit_per_trade']
            }
        }
        
        # Füge KI-Analyse hinzu
        analysis['ai_insights'] = self._get_ai_insights(results, strategy_config)
        
        return analysis

    def _is_better_strategy(self, results: BacktestResults) -> bool:
        """Prüft, ob die aktuelle Strategie besser ist als die bisherige beste"""
        if self.best_results is None:
            return True
            
        # Gewichte verschiedene Faktoren
        current_score = (
            results.sharpe_ratio * 0.3 +
            (1 - results.max_drawdown) * 0.2 +
            results.win_rate * 0.2 +
            results.profit_factor * 0.2 +
            (1 - results.risk_metrics['volatility']) * 0.1
        )
        
        best_score = (
            self.best_results.sharpe_ratio * 0.3 +
            (1 - self.best_results.max_drawdown) * 0.2 +
            self.best_results.win_rate * 0.2 +
            self.best_results.profit_factor * 0.2 +
            (1 - self.best_results.risk_metrics['volatility']) * 0.1
        )
        
        return current_score > best_score

    def _goals_achieved(self, results: BacktestResults) -> bool:
        """Prüft, ob alle Optimierungsziele erreicht wurden"""
        return all([
            results.sharpe_ratio >= self.target_metrics['min_sharpe_ratio'],
            results.max_drawdown <= self.target_metrics['max_drawdown_limit'],
            results.win_rate >= self.target_metrics['min_win_rate'],
            results.profit_factor >= self.target_metrics['min_profit_factor'],
            results.performance_summary['annualized_return'] >= self.target_metrics['min_annual_return'],
            results.risk_metrics['volatility'] <= self.target_metrics['max_volatility']
        ])

    def _ai_analyze_and_improve(self, iteration: int) -> None:
        """Lässt die KI die Ergebnisse analysieren und Verbesserungen vorschlagen"""
        try:
            # Bereite Daten für die Analyse vor
            current_results = self.optimization_history[-1]
            historical_data = self.optimization_history[:-1] if len(self.optimization_history) > 1 else []
            
            prompt = f"""Analysiere die Optimierungsergebnisse und schlage Verbesserungen vor.
            
            Aktuelle Ergebnisse:
            {json.dumps(current_results, indent=2)}
            
            Historische Optimierungen:
            {json.dumps(historical_data, indent=2)}
            
            Optimierungsziele:
            {json.dumps(self.target_metrics, indent=2)}
            
            Erstelle eine detaillierte Analyse mit:
            1. Hauptproblemen der aktuellen Strategie
            2. Erfolgreichen Aspekten, die beibehalten werden sollten
            3. Konkrete Verbesserungsvorschläge
            4. Empfehlungen für die nächste Iteration
            """
            
            analysis = self.model.analyze_optimization_results(prompt)
            
            # Speichere KI-Analyse
            self.optimization_history[-1]['ai_analysis'] = analysis
            
        except Exception as e:
            logger.error(f"Fehler bei der KI-Analyse: {str(e)}")

    def _get_ai_insights(self, results: BacktestResults, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert KI-basierte Einblicke in die Strategie-Performance"""
        try:
            prompt = f"""Analysiere die folgenden Backtest-Ergebnisse und generiere Einblicke:
            
            Performance Metriken:
            {json.dumps(results.performance_summary, indent=2)}
            
            Risiko Metriken:
            {json.dumps(results.risk_metrics, indent=2)}
            
            Strategie Konfiguration:
            {json.dumps(strategy_config, indent=2)}
            
            Generiere eine Analyse mit:
            1. Stärken und Schwächen der Strategie
            2. Potenzielle Verbesserungsbereiche
            3. Risikobewertung
            4. Langfristige Nachhaltigkeit
            """
            
            insights = self.model.analyze_strategy_performance(prompt)
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von KI-Einblicken: {str(e)}")
            return {}

    def _validate_strategy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und bereinigt die Strategiekonfiguration"""
        try:
            # Stelle sicher, dass alle erforderlichen Felder vorhanden sind
            required_fields = {
                'risk_management': {
                    'position_size_limit': 0.05,
                    'stop_loss_atr_multiple': 2.0,
                    'take_profit_atr_multiple': 3.0,
                    'max_positions': 10,
                    'sector_exposure_limit': 0.20
                },
                'entry_conditions': {
                    'min_volatility': 0.0,
                    'min_volume': 0.0,
                    'trend_strength': 0.0,
                    'correlation_threshold': 0.7
                },
                'exit_conditions': {
                    'profit_taking_threshold': 0.0,
                    'max_loss_threshold': 0.0,
                    'trend_reversal_threshold': 0.0
                },
                'timing': {
                    'holding_period': '3M',
                    'rebalancing_frequency': '1W',
                    'entry_timing_factors': []
                }
            }
            
            # Merge mit Standardwerten
            validated_config = self._merge_with_defaults(config, required_fields)
            
            # Validiere Wertebereiche
            validated_config = self._validate_value_ranges(validated_config)
            
            return validated_config
            
        except Exception as e:
            logger.error(f"Fehler bei der Konfigurationsvalidierung: {str(e)}")
            return self._get_default_strategy_config()

    def _merge_with_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Merged die Konfiguration mit Standardwerten"""
        merged = {}
        for key, default_value in defaults.items():
            if isinstance(default_value, dict):
                merged[key] = self._merge_with_defaults(
                    config.get(key, {}),
                    default_value
                )
            else:
                merged[key] = config.get(key, default_value)
        return merged

    def _validate_value_ranges(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert die Wertebereiche der Konfiguration"""
        # Risk Management
        config['risk_management']['position_size_limit'] = np.clip(
            config['risk_management']['position_size_limit'],
            0.01, 0.2
        )
        config['risk_management']['max_positions'] = int(np.clip(
            config['risk_management']['max_positions'],
            1, 50
        ))
        
        # Entry Conditions
        config['entry_conditions']['correlation_threshold'] = np.clip(
            config['entry_conditions']['correlation_threshold'],
            0.0, 1.0
        )
        
        # Exit Conditions
        config['exit_conditions']['max_loss_threshold'] = np.clip(
            config['exit_conditions']['max_loss_threshold'],
            0.01, 0.5
        )
        
        return config

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Liefert eine Standard-Strategiekonfiguration"""
        return {
            'risk_management': {
                'position_size_limit': 0.05,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'max_positions': 10,
                'sector_exposure_limit': 0.20
            },
            'entry_conditions': {
                'min_volatility': 0.01,
                'min_volume': 100000,
                'trend_strength': 0.6,
                'correlation_threshold': 0.7
            },
            'exit_conditions': {
                'profit_taking_threshold': 0.2,
                'max_loss_threshold': 0.1,
                'trend_reversal_threshold': 0.5
            },
            'timing': {
                'holding_period': '3M',
                'rebalancing_frequency': '1W',
                'entry_timing_factors': [
                    'trend_following',
                    'momentum',
                    'volatility_breakout'
                ]
            }
        } 