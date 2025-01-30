import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass

# Füge den src-Ordner zum Python-Path hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from risk.risk_manager import RiskManager
from models.deepseek_model import DeepseekAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from database.db_manager import DatabaseManager

logger = setup_logger(__name__)

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0  # Startkapital
    commission_rate: float = 0.001  # 0.1% Kommission pro Trade
    slippage: float = 0.001  # 0.1% Slippage
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    risk_free_rate: float = 0.02  # 2% risikofreier Zinssatz
    data_frequency: str = "1d"  # Datenfrequenz (1d, 1h, etc.)
    enable_short_selling: bool = False
    max_positions: int = 10
    rebalancing_frequency: str = "1w"  # Rebalancing-Frequenz
    benchmark_symbol: str = "^GDAXI"  # DAX als Benchmark

@dataclass
class BacktestResults:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[Dict[str, Any]]
    monthly_returns: pd.Series
    equity_curve: pd.Series
    benchmark_comparison: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    position_history: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    final_portfolio_value: float
    final_cash_position: float
    final_positions: Dict[str, Any]
    sector_allocation: Dict[str, float]

class Backtester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.risk_manager = RiskManager()
        self.market_analyzer = MarketAnalyzer()
        self.model = DeepseekAnalyzer()
        self.db_manager = DatabaseManager()
        
        # Initialisiere Tracking-Variablen
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.trades_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.portfolio_value: float = self.config.initial_capital
        self.cash: float = self.config.initial_capital
        
        # Performance-Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.config.initial_capital

    def load_market_data(self, data_path: str) -> pd.DataFrame:
        """Lädt historische Marktdaten"""
        try:
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {str(e)}")
            raise

    def load_news_data(self, news_path: str) -> pd.DataFrame:
        """Lädt historische Nachrichtendaten"""
        try:
            df = pd.read_csv(news_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Laden der Nachrichtendaten: {str(e)}")
            raise

    def run_backtest(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """Führt den Backtest durch"""
        logger.info("Starte Backtest")
        
        try:
            # Filtere Daten nach Zeitraum
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
            market_data = market_data.loc[start_date:end_date]
            
            if news_data is not None:
                news_data = news_data.loc[start_date:end_date]

            # Initialisiere Rebalancing-Zeitplan
            rebalancing_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq=self.config.rebalancing_frequency
            )

            # Hauptschleife des Backtests
            for current_date in market_data.index:
                # Aktualisiere Portfolio-Werte
                self._update_portfolio_values(market_data.loc[current_date])
                
                # Prüfe auf Rebalancing
                if current_date in rebalancing_dates:
                    self._perform_rebalancing(
                        market_data.loc[:current_date],
                        news_data.loc[:current_date] if news_data is not None else None
                    )
                
                # Prüfe Stop-Loss und Take-Profit
                self._check_position_limits(market_data.loc[current_date])
                
                # Tracke Performance
                self._track_performance(current_date)

            # Erstelle Backtest-Ergebnisse
            results = self._calculate_results(market_data)
            
            # Speichere Backtest-Ergebnisse in der Datenbank
            self.db_manager.save_backtest_results({
                'strategy_name': 'default_strategy',
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'trades': self.trades_history,
                'performance_summary': results.performance_summary,
                'risk_metrics': results.risk_metrics,
                'strategy_config': {}
            })
            
            # Speichere finalen Portfolio-Snapshot
            self.db_manager.save_portfolio_snapshot({
                'total_value': results.final_portfolio_value,
                'cash_position': results.final_cash_position,
                'positions': results.final_positions,
                'sector_allocation': results.sector_allocation,
                'risk_metrics': results.risk_metrics
            })
            
            logger.info("Backtest erfolgreich abgeschlossen")
            return results

        except Exception as e:
            logger.error(f"Fehler während des Backtests: {str(e)}")
            raise

    def _update_portfolio_values(self, current_prices: pd.Series) -> None:
        """Aktualisiert die Werte aller Positionen"""
        portfolio_value = self.cash
        
        for symbol, position in self.current_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                position['current_value'] = position_value
                position['unrealized_pnl'] = position_value - position['cost_basis']
                portfolio_value += position_value

        self.portfolio_value = portfolio_value
        self.market_analyzer.update_portfolio_value(portfolio_value)

    def _perform_rebalancing(
        self,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame]
    ) -> None:
        """Führt Portfolio-Rebalancing durch"""
        # Bereite Daten für Analyse vor
        market_data_dict = market_data.tail(30).to_dict()  # Letzte 30 Tage
        news_data_dict = news_data.tail(30).to_dict() if news_data is not None else None

        # Hole Analyseempfehlungen
        analysis = self.market_analyzer.analyze_data(
            market_data=json.dumps(market_data_dict),
            news_data=json.dumps(news_data_dict) if news_data_dict else None
        )

        # Verarbeite Handelsempfehlungen
        if "handelsempfehlung" in analysis:
            for empfehlung in analysis["handelsempfehlung"]:
                symbol = empfehlung.get("symbol")
                aktion = empfehlung.get("aktion")
                menge = empfehlung.get("menge")
                
                if symbol and aktion and menge:
                    self._execute_trade(
                        symbol=symbol,
                        aktion=aktion,
                        menge=menge,
                        preis=market_data.iloc[-1][symbol],
                        empfehlung=empfehlung
                    )

    def _execute_trade(
        self,
        symbol: str,
        aktion: str,
        menge: float,
        preis: float,
        empfehlung: Dict[str, Any]
    ) -> None:
        """Führt einen Trade aus"""
        try:
            # Berechne Handelskosten
            commission = preis * menge * self.config.commission_rate
            slippage = preis * menge * self.config.slippage
            total_cost = (preis * menge) + commission + slippage

            if aktion == "kaufen":
                if total_cost <= self.cash:
                    # Erstelle oder aktualisiere Position
                    if symbol not in self.current_positions:
                        self.current_positions[symbol] = {
                            'quantity': menge,
                            'cost_basis': total_cost,
                            'entry_price': preis,
                            'entry_date': datetime.now(),
                            'stop_loss': empfehlung.get('stop_loss'),
                            'take_profit': empfehlung.get('take_profit')
                        }
                    else:
                        # Erhöhe bestehende Position
                        position = self.current_positions[symbol]
                        new_quantity = position['quantity'] + menge
                        new_cost_basis = position['cost_basis'] + total_cost
                        position.update({
                            'quantity': new_quantity,
                            'cost_basis': new_cost_basis,
                            'average_price': new_cost_basis / new_quantity
                        })

                    self.cash -= total_cost
                    self._log_trade("BUY", symbol, menge, preis, commission, slippage)

            elif aktion == "verkaufen":
                if symbol in self.current_positions:
                    position = self.current_positions[symbol]
                    if menge >= position['quantity']:  # Vollständiger Verkauf
                        verkaufs_erloes = (preis * position['quantity']) - commission - slippage
                        self.cash += verkaufs_erloes
                        realized_pnl = verkaufs_erloes - position['cost_basis']
                        
                        self._log_trade("SELL", symbol, position['quantity'], preis, commission, slippage, realized_pnl)
                        del self.current_positions[symbol]
                    else:  # Teilverkauf
                        verkaufs_erloes = (preis * menge) - commission - slippage
                        self.cash += verkaufs_erloes
                        
                        # Aktualisiere Position
                        cost_basis_per_unit = position['cost_basis'] / position['quantity']
                        realized_pnl = verkaufs_erloes - (cost_basis_per_unit * menge)
                        
                        position.update({
                            'quantity': position['quantity'] - menge,
                            'cost_basis': position['cost_basis'] - (cost_basis_per_unit * menge)
                        })
                        
                        self._log_trade("PARTIAL_SELL", symbol, menge, preis, commission, slippage, realized_pnl)

        except Exception as e:
            logger.error(f"Fehler bei Trade-Ausführung: {str(e)}")

    def _check_position_limits(self, current_prices: pd.Series) -> None:
        """Überprüft Stop-Loss und Take-Profit Limits"""
        for symbol, position in list(self.current_positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Prüfe Stop-Loss
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    self._execute_trade(
                        symbol=symbol,
                        aktion="verkaufen",
                        menge=position['quantity'],
                        preis=current_price,
                        empfehlung={"typ": "stop_loss"}
                    )
                
                # Prüfe Take-Profit
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    self._execute_trade(
                        symbol=symbol,
                        aktion="verkaufen",
                        menge=position['quantity'],
                        preis=current_price,
                        empfehlung={"typ": "take_profit"}
                    )

    def _track_performance(self, current_date: datetime) -> None:
        """Trackt die Performance des Portfolios"""
        self.equity_curve.append((current_date, self.portfolio_value))
        
        # Aktualisiere Drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        else:
            current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def _log_trade(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
        pnl: float = 0.0
    ) -> None:
        """Protokolliert einen ausgeführten Trade"""
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'slippage': slippage,
            'pnl': pnl,
            'portfolio_value': self.portfolio_value
        }
        
        self.trades_history.append(trade)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        elif pnl < 0:
            self.total_loss += abs(pnl)

    def _calculate_results(self, market_data: pd.DataFrame) -> BacktestResults:
        """Berechnet die finalen Backtest-Ergebnisse"""
        # Erstelle Equity Curve als pandas Series
        equity_curve = pd.Series(
            [value for _, value in self.equity_curve],
            index=[date for date, _ in self.equity_curve]
        )
        
        # Berechne monatliche Returns
        monthly_returns = equity_curve.resample('M').last().pct_change()
        
        # Berechne Benchmark-Performance
        benchmark_returns = market_data[self.config.benchmark_symbol].pct_change()
        
        # Berechne Performance-Metriken
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self.total_profit / abs(self.total_loss) if self.total_loss != 0 else float('inf')
        
        # Berechne Sharpe Ratio
        excess_returns = monthly_returns - (self.config.risk_free_rate / 12)
        sharpe_ratio = np.sqrt(12) * (excess_returns.mean() / excess_returns.std()) if len(excess_returns) > 1 else 0
        
        # Erstelle Performance Summary
        performance_summary = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / len(equity_curve)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': self.total_trades,
            'avg_trade_duration': self._calculate_avg_trade_duration(),
            'avg_profit_per_trade': (self.total_profit - abs(self.total_loss)) / self.total_trades if self.total_trades > 0 else 0
        }
        
        # Erstelle Risiko-Metriken
        risk_metrics = {
            'volatility': monthly_returns.std() * np.sqrt(12),
            'var_95': monthly_returns.quantile(0.05),
            'var_99': monthly_returns.quantile(0.01),
            'beta': self._calculate_beta(monthly_returns, benchmark_returns),
            'alpha': self._calculate_alpha(monthly_returns, benchmark_returns),
            'sortino_ratio': self._calculate_sortino_ratio(monthly_returns),
            'calmar_ratio': abs(total_return / self.max_drawdown) if self.max_drawdown != 0 else float('inf')
        }
        
        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=self.trades_history,
            monthly_returns=monthly_returns,
            equity_curve=equity_curve,
            benchmark_comparison=self._compare_to_benchmark(equity_curve, market_data),
            risk_metrics=risk_metrics,
            position_history=self._get_position_history(),
            performance_summary=performance_summary,
            final_portfolio_value=self.portfolio_value,
            final_cash_position=self.cash,
            final_positions=self.current_positions,
            sector_allocation=self._calculate_sector_allocation()
        )

    def _calculate_avg_trade_duration(self) -> float:
        """Berechnet die durchschnittliche Handelsdauer"""
        if not self.trades_history:
            return 0.0
            
        durations = []
        open_trades = {}
        
        for trade in self.trades_history:
            if trade['action'] in ['BUY', 'PARTIAL_BUY']:
                if trade['symbol'] not in open_trades:
                    open_trades[trade['symbol']] = trade['timestamp']
            elif trade['action'] in ['SELL', 'PARTIAL_SELL']:
                if trade['symbol'] in open_trades:
                    duration = (trade['timestamp'] - open_trades[trade['symbol']]).total_seconds() / 86400  # Tage
                    durations.append(duration)
                    if trade['action'] == 'SELL':
                        del open_trades[trade['symbol']]
        
        return np.mean(durations) if durations else 0.0

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Berechnet das Beta des Portfolios"""
        if len(returns) <= 1:
            return 0.0
        
        # Bereinige die Daten
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) <= 1:
            return 0.0
            
        covariance = aligned_data.cov().iloc[0, 1]
        benchmark_variance = benchmark_returns.var()
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 0.0

    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Berechnet das Alpha des Portfolios"""
        if len(returns) <= 1:
            return 0.0
            
        beta = self._calculate_beta(returns, benchmark_returns)
        excess_return = returns.mean() - self.config.risk_free_rate / 12
        benchmark_excess_return = benchmark_returns.mean() - self.config.risk_free_rate / 12
        
        return excess_return - (beta * benchmark_excess_return)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Berechnet das Sortino Ratio"""
        if len(returns) <= 1:
            return 0.0
            
        excess_returns = returns - (self.config.risk_free_rate / 12)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        return np.sqrt(12) * (excess_returns.mean() / downside_std) if downside_std != 0 else 0.0

    def _compare_to_benchmark(self, equity_curve: pd.Series, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Vergleicht die Performance mit dem Benchmark"""
        benchmark_prices = market_data[self.config.benchmark_symbol]
        benchmark_returns = benchmark_prices.pct_change()
        
        # Normalisiere beide Kurven auf 100
        equity_curve_norm = 100 * (1 + equity_curve.pct_change()).cumprod()
        benchmark_norm = 100 * (1 + benchmark_returns).cumprod()
        
        return {
            'strategy_final_value': equity_curve_norm.iloc[-1],
            'benchmark_final_value': benchmark_norm.iloc[-1],
            'outperformance': equity_curve_norm.iloc[-1] - benchmark_norm.iloc[-1],
            'correlation': equity_curve_norm.corr(benchmark_norm),
            'tracking_error': (equity_curve_norm - benchmark_norm).std() * np.sqrt(252),
            'information_ratio': (equity_curve_norm - benchmark_norm).mean() / (equity_curve_norm - benchmark_norm).std() if len(equity_curve_norm) > 1 else 0
        }

    def _get_position_history(self) -> List[Dict[str, Any]]:
        """Erstellt eine Historie aller Positionsänderungen"""
        position_history = []
        
        for trade in self.trades_history:
            position_snapshot = {
                'timestamp': trade['timestamp'],
                'portfolio_value': trade['portfolio_value'],
                'cash': self.cash,
                'positions': {
                    symbol: {
                        'quantity': pos['quantity'],
                        'value': pos['current_value'],
                        'unrealized_pnl': pos['unrealized_pnl']
                    }
                    for symbol, pos in self.current_positions.items()
                }
            }
            position_history.append(position_snapshot)
        
        return position_history

    def _calculate_sector_allocation(self) -> Dict[str, float]:
        """Berechnet die aktuelle Sektorallokation"""
        # Implementiere hier die Sektorallokationsberechnung
        return {}

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Berechnet aktuelle Risikometriken"""
        # Implementiere hier die Risikoberechnung
        return {} 