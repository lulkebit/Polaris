from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from models.deepseek_model import DeepseekAnalyzer
from backtesting.backtester import Backtester, BacktestConfig, BacktestResults

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    max_iterations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    optimization_metric: str = "sharpe_ratio"  # oder "total_return", "sortino_ratio", etc.
    min_trades: int = 20
    min_win_rate: float = 0.4
    max_drawdown_limit: float = 0.2
    min_profit_factor: float = 1.2

class StrategyOptimizer:
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.model = DeepseekAnalyzer()
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_strategy: Optional[Dict[str, Any]] = None
        self.best_results: Optional[BacktestResults] = None
        self.target_metrics = {
            "min_sharpe_ratio": 1.0,
            "max_drawdown_limit": 0.15,
            "min_win_rate": 0.55,
            "min_profit_factor": 1.5
        }
        
    def optimize_strategy(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
        max_iterations: int = 10,
        optimization_period: str = "1Y"
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Führt eine KI-gesteuerte Strategieoptimierung durch"""
        
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
                raise ValueError("Keine gültige Strategie gefunden")
                
            return self.best_strategy, self.best_results
            
        except Exception as e:
            logger.error(f"Fehler bei der Strategieoptimierung: {str(e)}")
            raise

    def _generate_strategy_config(self, iteration: int) -> Dict[str, Any]:
        """Generiert eine neue Strategiekonfiguration"""
        if iteration == 0 or not self.optimization_history:
            # Erste Iteration: Verwende Standardkonfiguration
            return self._get_default_strategy_config()
        
        # Analysiere bisherige Ergebnisse
        previous_results = [
            {
                'config': hist['strategy_config'],
                'performance': self._calculate_fitness(hist['results'])
            }
            for hist in self.optimization_history
        ]
        
        # Sortiere nach Performance
        previous_results.sort(key=lambda x: x['performance'], reverse=True)
        
        # Wähle die besten Konfigurationen für Kreuzung
        top_configs = [r['config'] for r in previous_results[:2]]
        
        # Generiere neue Konfiguration durch Kreuzung und Mutation
        new_config = self._crossover_configs(top_configs[0], top_configs[1])
        new_config = self._mutate_config(new_config)
        
        return self._validate_strategy_config(new_config)

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
                    'value': results.performance_summary['sharpe_ratio'],
                    'target_achieved': results.performance_summary['sharpe_ratio'] >= self.target_metrics['min_sharpe_ratio']
                },
                'max_drawdown': {
                    'value': results.performance_summary['max_drawdown'],
                    'target_achieved': results.performance_summary['max_drawdown'] <= self.target_metrics['max_drawdown_limit']
                },
                'win_rate': {
                    'value': results.performance_summary['win_rate'],
                    'target_achieved': results.performance_summary['win_rate'] >= self.target_metrics['min_win_rate']
                },
                'profit_factor': {
                    'value': results.performance_summary['profit_factor'],
                    'target_achieved': results.performance_summary['profit_factor'] >= self.target_metrics['min_profit_factor']
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
        """Prüft, ob die neue Strategie besser ist als die bisherige beste"""
        if self.best_results is None:
            return True
            
        # Vergleiche basierend auf der konfigurierten Optimierungsmetrik
        if self.config.optimization_metric == "sharpe_ratio":
            return results.performance_summary['sharpe_ratio'] > self.best_results.performance_summary['sharpe_ratio']
        elif self.config.optimization_metric == "total_return":
            return results.performance_summary['total_return'] > self.best_results.performance_summary['total_return']
        elif self.config.optimization_metric == "sortino_ratio":
            return results.risk_metrics['sortino_ratio'] > self.best_results.risk_metrics['sortino_ratio']
        
        return False

    def _goals_achieved(self, results: BacktestResults) -> bool:
        """Prüft, ob alle Optimierungsziele erreicht wurden"""
        return all([
            results.performance_summary['sharpe_ratio'] >= self.target_metrics['min_sharpe_ratio'],
            results.performance_summary['max_drawdown'] <= self.target_metrics['max_drawdown_limit'],
            results.performance_summary['win_rate'] >= self.target_metrics['min_win_rate'],
            results.performance_summary['profit_factor'] >= self.target_metrics['min_profit_factor'],
            len(results.trades) >= self.config.min_trades
        ])

    def _calculate_fitness(self, results: BacktestResults) -> float:
        """Berechnet den Fitness-Wert einer Strategie"""
        metrics = results.performance_summary
        
        # Gewichte für verschiedene Metriken
        weights = {
            'sharpe_ratio': 0.3,
            'sortino_ratio': 0.2,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'max_drawdown': 0.2
        }
        
        # Normalisiere Metriken
        normalized_metrics = {
            'sharpe_ratio': max(0, min(1, metrics['sharpe_ratio'] / 3)),
            'sortino_ratio': max(0, min(1, metrics['sortino_ratio'] / 3)),
            'win_rate': metrics['win_rate'],
            'profit_factor': max(0, min(1, (metrics['profit_factor'] - 1) / 2)),
            'max_drawdown': max(0, 1 - metrics['max_drawdown'] / 0.4)
        }
        
        # Berechne gewichteten Fitness-Wert
        fitness = sum(
            normalized_metrics[metric] * weight
            for metric, weight in weights.items()
        )
        
        return fitness

    def _crossover_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Führt eine Kreuzung zwischen zwei Strategiekonfigurationen durch"""
        new_config = {}
        
        for key in config1.keys():
            if isinstance(config1[key], dict):
                new_config[key] = self._crossover_configs(config1[key], config2[key])
            else:
                # Wähle zufällig zwischen den Werten oder berechne den Durchschnitt
                if np.random.random() < self.config.crossover_rate:
                    new_config[key] = config1[key]
                else:
                    if isinstance(config1[key], (int, float)):
                        new_config[key] = (config1[key] + config2[key]) / 2
                    else:
                        new_config[key] = config2[key]
        
        return new_config

    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Mutationen in der Strategiekonfiguration durch"""
        mutated_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                mutated_config[key] = self._mutate_config(value)
            else:
                if np.random.random() < self.config.mutation_rate:
                    if isinstance(value, float):
                        # Mutiere Float-Werte um ±20%
                        mutation_factor = 1 + (np.random.random() - 0.5) * 0.4
                        mutated_config[key] = value * mutation_factor
                    elif isinstance(value, int):
                        # Mutiere Integer-Werte um ±2
                        mutation = np.random.randint(-2, 3)
                        mutated_config[key] = max(1, value + mutation)
                    else:
                        mutated_config[key] = value
                else:
                    mutated_config[key] = value
        
        return mutated_config

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

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Liefert die Standard-Strategiekonfiguration"""
        return {
            'risk_management': {
                'position_size_limit': 0.05,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'max_positions': 10,
                'sector_exposure_limit': 0.20
            },
            'entry_conditions': {
                'min_volatility': 0.10,
                'min_volume': 100000,
                'trend_strength': 0.6,
                'correlation_threshold': 0.7
            },
            'exit_conditions': {
                'profit_taking_threshold': 0.15,
                'max_loss_threshold': 0.10,
                'trend_reversal_threshold': 0.5
            },
            'timing': {
                'holding_period': '3M',
                'rebalancing_frequency': '1W',
                'entry_timing_factors': [
                    'momentum',
                    'volatility',
                    'volume'
                ]
            }
        }

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

    def _ai_analyze_and_improve(self, iteration: int) -> None:
        """Lässt die KI die Ergebnisse analysieren und Verbesserungen vorschlagen"""
        if not self.optimization_history:
            return
            
        try:
            # Bereite Daten für die Analyse vor
            current_results = self.optimization_history[-1]
            
            prompt = f"""Analysiere die Optimierungsergebnisse der Iteration {iteration}:
            
            Performance:
            {json.dumps(current_results['results'].performance_summary, indent=2)}
            
            Konfiguration:
            {json.dumps(current_results['strategy_config'], indent=2)}
            
            Analyse:
            {json.dumps(current_results['analysis'], indent=2)}
            
            Schlage Verbesserungen vor für:
            1. Risk Management Parameter
            2. Entry/Exit Bedingungen
            3. Timing Parameter
            """
            
            # Hole KI-Vorschläge
            improvements = self.model.get_strategy_improvements(prompt)
            
            # Speichere Vorschläge in der Historie
            self.optimization_history[-1]['ai_improvements'] = improvements
            
        except Exception as e:
            logger.error(f"Fehler bei der KI-Analyse: {str(e)}") 