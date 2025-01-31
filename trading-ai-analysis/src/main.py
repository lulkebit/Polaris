from datetime import datetime
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from storage.database import DatabaseConnection
from utils.data_processor import calculate_technical_indicators
from models.strategy import MeanReversionStrategy
from backtesting import Backtest
import backtrader as bt
from models.risk_management import RiskManager
from utils.ai_logger import AILogger
from utils.console_logger import ConsoleLogger
from sqlalchemy import text

class AnalysisPipeline:
    def __init__(self):
        self.db = DatabaseConnection()
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-base",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        self.risk_manager = RiskManager()
        
        # Cerebro-Instanz für Backtrader initialisieren
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        
        # Strategie zu Cerebro hinzufügen
        self.cerebro.addstrategy(MeanReversionStrategy, risk_manager=self.risk_manager)
        
        self.logger = AILogger(name="analysis_pipeline")
        self.console = ConsoleLogger(name="trading_console", log_file="logs/console.log")

    def run_analysis(self):
        """Hauptworkflow für Datenanalyse und Strategieausführung"""
        start_time = datetime.now()
        self.console.section("Trading AI Analysis Pipeline")
        
        try:
            self.logger.log_model_metrics(
                model_name="analysis_pipeline",
                metrics={"status": "started"}
            )
            self.console.info("Pipeline gestartet")
            
            # 0. Datenbank-Initialisierung prüfen
            self.console.info("Initialisiere Datenbankverbindung...")
            self._check_and_initialize_database()
            self.console.success("Datenbankverbindung hergestellt")
            
            # 1. Daten aus der Pipeline laden
            self.console.info("Lade Daten aus der Pipeline...")
            raw_data = self._load_pipeline_data()
            self.logger.log_model_metrics(
                model_name="data_loading",
                metrics={
                    "rows_loaded": len(raw_data),
                    "symbols": len(raw_data['symbol'].unique())
                }
            )
            self.console.success(f"Daten geladen: {len(raw_data)} Zeilen für {len(raw_data['symbol'].unique())} Symbole")
            
            # 2. Datenaufbereitung und Feature-Engineering
            self.console.info("Führe Feature-Engineering durch...")
            processed_data = self._preprocess_data(raw_data)
            self.logger.log_model_metrics(
                model_name="preprocessing",
                metrics={
                    "processed_rows": len(processed_data),
                    "features_generated": len(processed_data.columns)
                }
            )
            self.console.success(f"Features generiert: {len(processed_data.columns)} Features")
            
            # 3. Risiko-Check vor der Analyse
            self.console.info("Führe Risiko-Checks durch...")
            if not self._pre_risk_checks(processed_data):
                self.logger.log_model_metrics(
                    model_name="risk_check",
                    metrics={"status": "failed"}
                )
                self.console.failure("Risiko-Checks fehlgeschlagen - Analyse abgebrochen")
                return
            self.console.success("Risiko-Checks bestanden")
                
            # 4. KI-basierte Analyse
            self.console.section("KI-Analyse")
            total_symbols = len(processed_data['symbol'].unique())
            predictions = pd.DataFrame()
            
            for idx, symbol in enumerate(processed_data['symbol'].unique(), 1):
                symbol_data = processed_data[processed_data['symbol'] == symbol]
                self.console.progress(idx, total_symbols, f"Analysiere {symbol}")
                
                # Modellvorhersagen
                features = self._extract_features(symbol_data)
                inputs = self.tokenizer(str(features), return_tensors="pt", truncation=False).to(self.model.device)
                outputs = self.model.generate(**inputs, max_new_tokens=20000, pad_token_id=self.tokenizer.eos_token_id)
                prediction = float(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
                confidence = self._calculate_confidence(prediction)
                
                self.logger.log_prediction(
                    symbol=symbol,
                    prediction=prediction,
                    confidence=confidence,
                    features=features
                )
                
                predictions = predictions.append({
                    'symbol': symbol,
                    'prediction': prediction,
                    'confidence': confidence
                }, ignore_index=True)
            
            self.console.success(f"KI-Analyse abgeschlossen für {total_symbols} Symbole")
            
            # 5. Signalgenerierung
            self.console.section("Signal-Generierung")
            signals = self._generate_signals(processed_data, predictions)
            buy_signals = len(signals[signals['signal'] > 0])
            sell_signals = len(signals[signals['signal'] < 0])
            
            self.logger.log_model_metrics(
                model_name="signals",
                metrics={
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals
                }
            )
            self.console.info(f"Generierte Signale: {buy_signals} Kauf, {sell_signals} Verkauf")
            
            # 6. Backtesting und Optimierung
            self.console.section("Backtesting")
            backtest_results = self._run_backtesting(processed_data)
            
            for strategy_name, result in backtest_results.items():
                self.console.info(f"\nErgebnisse für {strategy_name}:")
                self.console.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                self.console.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
                self.console.info(f"Gesamtrendite: {result['total_return']:.2%}")
            
            # 7. Ergebnisse speichern
            self.console.info("Speichere Ergebnisse...")
            self._save_results(signals, backtest_results)
            
            self.logger.log_model_metrics(
                model_name="analysis_pipeline",
                metrics={"status": "completed"}
            )
            
            self.console.section("Analyse abgeschlossen")
            self.console.timing(start_time)
            
        except Exception as e:
            self.logger.log_model_metrics(
                model_name="analysis_pipeline",
                metrics={
                    "status": "error",
                    "error_message": str(e)
                }
            )
            self.console.failure(f"Fehler in der Pipeline: {str(e)}")
            raise

    def _check_and_initialize_database(self):
        """Überprüft die Datenbankverbindung"""
        try:
            # Teste die Verbindung durch Ausführung einer einfachen Query
            with self.db.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                self.logger.log_model_metrics(
                    model_name="database",
                    metrics={"status": "connected"}
                )
        except Exception as e:
            self.logger.log_model_metrics(
                model_name="database",
                metrics={
                    "status": "error",
                    "error_message": str(e)
                }
            )
            raise

    def _load_pipeline_data(self):
        """Lädt die Daten aus der Pipeline"""
        data = self.db.get_latest_data()
        return data

    def _preprocess_data(self, data):
        """Bereitet die Daten auf und führt Feature-Engineering durch"""
        return calculate_technical_indicators(data)

    def _pre_risk_checks(self, data):
        """Führt Risikochecks vor der Analyse durch (deaktiviert)"""
        self.logger.log_model_metrics(
            model_name="risk_metrics",
            metrics={
                "risk_checks_disabled": True,
                "status": "passed"
            }
        )
        
        self.console.info("Risiko-Checks sind deaktiviert")
        return True

    def _extract_features(self, data):
        """Extrahiert Features für das KI-Modell"""
        features = {
            'price_stats': {
                'close_mean': float(data['close'].mean()),
                'close_std': float(data['close'].std()),
                'volume_mean': float(data['volume'].mean()),
                'high_max': float(data['high'].max()),
                'low_min': float(data['low'].min())
            }
        }
        
        # Technische Features hinzufügen, falls vorhanden
        tech_cols = data.filter(like='technical_').columns
        if len(tech_cols) > 0:
            features['technical_stats'] = data[tech_cols].agg(['mean', 'std']).to_dict()
            
        return features

    def _calculate_confidence(self, prediction):
        """Berechnet den Konfidenzwert der Vorhersage"""
        # Einfache Sigmoid-Funktion für Konfidenzwerte zwischen 0 und 1
        confidence = 1 / (1 + np.exp(-abs(prediction)))
        return float(confidence)

    def _generate_predictions(self, data):
        """Generiert Vorhersagen mit dem KI-Modell"""
        predictions = pd.DataFrame()
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            # Modellvorhersagen
            features = self._extract_features(symbol_data)
            inputs = self.tokenizer(str(features), return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_length=100)
            prediction = float(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            confidence = self._calculate_confidence(prediction)
            
            self.logger.log_prediction(
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                features=features
            )
            
            predictions = predictions.append({
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence
            }, ignore_index=True)
        
        return predictions

    def _generate_signals(self, data, predictions):
        """Generiert Handelssignale basierend auf Vorhersagen"""
        signals = pd.DataFrame()
        for symbol in data['symbol'].unique():
            symbol_pred = predictions[predictions['symbol'] == symbol].iloc[-1]
            
            signal = 1 if symbol_pred['prediction'] > 0.7 else (-1 if symbol_pred['prediction'] < 0.3 else 0)
            
            self.logger.log_prediction(
                symbol=symbol,
                prediction=signal,
                confidence=symbol_pred['confidence'],
                metadata={"signal_type": "trading_signal"}
            )
            
            signals = signals.append({
                'symbol': symbol,
                'signal': signal,
                'confidence': symbol_pred['confidence']
            }, ignore_index=True)
        
        return signals

    def _run_backtesting(self, data):
        """Führt Backtesting für die Strategien durch"""
        results = {}
        for name, strategy in self.strategies.items():
            backtest = Backtest(data, strategy)
            result = backtest.run()
            
            self.logger.log_model_metrics(
                model_name=f"backtest_{name}",
                metrics=result
            )
            
            results[name] = result
        
        return results

    def _save_results(self, signals, backtest_results):
        """Speichert die Analyseergebnisse"""
        self.db.save_signals(signals)
        self.db.save_backtest_results(backtest_results)
        
        self.logger.log_model_metrics(
            model_name="results_saving",
            metrics={
                "signals_saved": len(signals),
                "backtest_results_saved": len(backtest_results)
            }
        )

if __name__ == "__main__":
    pipeline = AnalysisPipeline()
    pipeline.run_analysis() 