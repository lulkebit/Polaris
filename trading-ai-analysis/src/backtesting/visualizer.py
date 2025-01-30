import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from .backtester import BacktestResults
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    monthly_returns: pd.Series
    performance_summary: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    position_history: List[Dict[str, Any]]
    benchmark_comparison: Dict[str, Any]

class BacktestVisualizer:
    def __init__(self, results: BacktestResults):
        self.results = results
        self.setup_style()

    def setup_style(self) -> None:
        """Konfiguriert den Plot-Stil"""
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def create_performance_dashboard(self, output_path: str) -> None:
        """Erstellt ein umfassendes Performance-Dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve vs Benchmark',
                'Monatliche Returns',
                'Drawdown Analyse',
                'Position Sizing',
                'Trade Performance',
                'Risk Metrics'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )

        # Equity Curve vs Benchmark
        fig.add_trace(
            go.Scatter(
                x=self.results.equity_curve.index,
                y=self.results.equity_curve.values,
                name='Strategy',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if 'benchmark_values' in self.results.benchmark_comparison:
            fig.add_trace(
                go.Scatter(
                    x=self.results.benchmark_comparison['benchmark_values'].index,
                    y=self.results.benchmark_comparison['benchmark_values'].values,
                    name='Benchmark',
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )

        # Monatliche Returns
        fig.add_trace(
            go.Bar(
                x=self.results.monthly_returns.index,
                y=self.results.monthly_returns.values * 100,
                name='Monthly Returns',
                marker_color='green'
            ),
            row=1, col=2
        )

        # Drawdown Analyse
        drawdown = self._calculate_drawdown_series()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # Position Sizing
        position_sizes = self._get_position_sizes()
        fig.add_trace(
            go.Scatter(
                x=position_sizes.index,
                y=position_sizes.values,
                name='Position Size',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

        # Performance Tabelle
        performance_table = self._create_performance_table()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(performance_table.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[performance_table[col] for col in performance_table.columns],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=1
        )

        # Risiko Metriken Tabelle
        risk_table = self._create_risk_table()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(risk_table.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[risk_table[col] for col in risk_table.columns],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=2
        )

        # Update Layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Backtest Performance Dashboard",
            showlegend=True
        )

        # Speichere Dashboard
        fig.write_html(output_path)

    def plot_trade_analysis(self, output_path: str) -> None:
        """Erstellt eine detaillierte Analyse der Trades"""
        trades_df = pd.DataFrame(self.results.trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Profit/Loss pro Trade',
                'Trade Duration Distribution',
                'Cumulative PnL',
                'Win/Loss Ratio by Month'
            )
        )

        # PnL pro Trade
        fig.add_trace(
            go.Bar(
                x=trades_df.index,
                y=trades_df['pnl'],
                name='Trade PnL',
                marker_color=trades_df['pnl'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                )
            ),
            row=1, col=1
        )

        # Trade Duration Distribution
        durations = self._calculate_trade_durations(trades_df)
        fig.add_trace(
            go.Histogram(
                x=durations,
                name='Trade Durations',
                nbinsx=20,
                marker_color='blue'
            ),
            row=1, col=2
        )

        # Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades_df.index,
                y=cumulative_pnl,
                name='Cumulative PnL',
                line=dict(color='green')
            ),
            row=2, col=1
        )

        # Win/Loss Ratio by Month
        monthly_stats = self._calculate_monthly_win_loss(trades_df)
        fig.add_trace(
            go.Bar(
                x=monthly_stats.index.astype(str),
                y=monthly_stats['win_rate'] * 100,
                name='Win Rate',
                marker_color='blue'
            ),
            row=2, col=2
        )

        # Update Layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Detailed Trade Analysis",
            showlegend=True
        )

        # Speichere Plot
        fig.write_html(output_path)

    def create_risk_report(self, output_path: str) -> None:
        """Erstellt einen detaillierten Risikobericht"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Metrics Over Time',
                'Value at Risk Analysis',
                'Position Correlation Matrix',
                'Sector Exposure'
            )
        )

        # Risk Metrics Over Time
        risk_metrics = self._calculate_rolling_risk_metrics()
        for metric in risk_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics.index,
                    y=risk_metrics[metric],
                    name=metric,
                    line=dict(width=1)
                ),
                row=1, col=1
            )

        # VaR Analysis
        returns = self.results.equity_curve.pct_change().dropna()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns Distribution',
                nbinsx=50,
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        # Füge VaR-Linien hinzu
        fig.add_vline(x=var_95, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", row=1, col=2)

        # Position Correlation Matrix
        position_correlations = self._calculate_position_correlations()
        fig.add_trace(
            go.Heatmap(
                z=position_correlations.values,
                x=position_correlations.columns,
                y=position_correlations.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ),
            row=2, col=1
        )

        # Sector Exposure
        sector_exposure = self._calculate_sector_exposure()
        fig.add_trace(
            go.Pie(
                labels=list(sector_exposure.keys()),
                values=list(sector_exposure.values()),
                name='Sector Exposure'
            ),
            row=2, col=2
        )

        # Update Layout
        fig.update_layout(
            height=1000,
            width=1400,
            title_text="Risk Analysis Report",
            showlegend=True
        )

        # Speichere Report
        fig.write_html(output_path)

    def _calculate_drawdown_series(self) -> pd.Series:
        """Berechnet die Drawdown-Serie"""
        equity = self.results.equity_curve
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown

    def _get_position_sizes(self) -> pd.Series:
        """Extrahiert die Positionsgrößen über Zeit"""
        position_sizes = pd.DataFrame(self.results.position_history)
        position_sizes['total_position_value'] = position_sizes['positions'].apply(
            lambda x: sum(pos['value'] for pos in x.values())
        )
        return pd.Series(
            position_sizes['total_position_value'].values,
            index=pd.to_datetime(position_sizes['timestamp'])
        )

    def _create_performance_table(self) -> pd.DataFrame:
        """Erstellt eine Tabelle mit Performance-Metriken"""
        metrics = self.results.performance_summary
        return pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Sharpe Ratio',
                'Max Drawdown',
                'Win Rate',
                'Profit Factor'
            ],
            'Value': [
                f"{metrics['total_return']*100:.2f}%",
                f"{metrics['annualized_return']*100:.2f}%",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['max_drawdown']*100:.2f}%",
                f"{metrics['win_rate']*100:.2f}%",
                f"{metrics['profit_factor']:.2f}"
            ]
        })

    def _create_risk_table(self) -> pd.DataFrame:
        """Erstellt eine Tabelle mit Risiko-Metriken"""
        metrics = self.results.risk_metrics
        return pd.DataFrame({
            'Metric': [
                'Volatility',
                'VaR (95%)',
                'VaR (99%)',
                'Beta',
                'Alpha',
                'Sortino Ratio'
            ],
            'Value': [
                f"{metrics['volatility']*100:.2f}%",
                f"{metrics['var_95']*100:.2f}%",
                f"{metrics['var_99']*100:.2f}%",
                f"{metrics['beta']:.2f}",
                f"{metrics['alpha']*100:.2f}%",
                f"{metrics['sortino_ratio']:.2f}"
            ]
        })

    def _calculate_trade_durations(self, trades_df: pd.DataFrame) -> pd.Series:
        """Berechnet die Handelsdauer für jeden Trade"""
        trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
        return trades_df['duration'].dt.total_seconds() / (24 * 60 * 60)  # Konvertiere zu Tagen

    def _calculate_monthly_win_loss(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet monatliche Gewinn/Verlust-Statistiken"""
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_stats = trades_df.groupby('month').agg({
            'pnl': ['count', lambda x: (x > 0).mean()]
        })
        monthly_stats.columns = ['trade_count', 'win_rate']
        return monthly_stats

    def _calculate_rolling_risk_metrics(self, window: int = 20) -> pd.DataFrame:
        """Berechnet rollende Risikometriken"""
        returns = self.results.equity_curve.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame({
            'Volatility': returns.rolling(window).std() * np.sqrt(252),
            'Sharpe': returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252),
            'Sortino': returns.rolling(window).mean() / returns[returns < 0].rolling(window).std() * np.sqrt(252)
        })
        
        return rolling_metrics

    def _calculate_position_correlations(self) -> pd.DataFrame:
        """Berechnet Korrelationen zwischen Positionen"""
        position_values = pd.DataFrame()
        
        for snapshot in self.results.position_history:
            date = pd.to_datetime(snapshot['timestamp'])
            positions = snapshot['positions']
            
            for symbol, data in positions.items():
                position_values.loc[date, symbol] = data['value']
                
        return position_values.corr()

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Berechnet die aktuelle Sektorexposition"""
        sector_exposure = {}
        latest_positions = self.results.position_history[-1]['positions']
        total_value = sum(pos['value'] for pos in latest_positions.values())
        
        for position in latest_positions.values():
            sector = position.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position['value'] / total_value
        
        return sector_exposure 