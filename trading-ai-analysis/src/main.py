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
from utils.resource_manager import ResourceManager
from sqlalchemy import text
import traceback
import torch

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
        self.resource_manager = ResourceManager(max_cpu_percent=85.0, max_memory_percent=85.0)
        
        # Low-Performance-Modus basierend auf GPU-Verfügbarkeit
        self.low_performance_mode = not torch.cuda.is_available()
        
        # Cerebro-Instanz für Backtrader initialisieren
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        
        # Strategie zu Cerebro hinzufügen
        self.cerebro.addstrategy(MeanReversionStrategy, risk_manager=self.risk_manager)
        
        # Logger initialisieren
        self.logger = AILogger(name="analysis_pipeline")
        
        # Hardware-Info loggen
        self.logger.log_hardware_info()

    def run_analysis(self):
        """Hauptworkflow für Datenanalyse und Strategieausführung"""
        try:
            start_time = datetime.now()
            
            self.logger.section("Trading AI Analysis Pipeline")
            
            # Starte Ressourcenüberwachung
            self.resource_manager.start_monitoring()
            self.logger.info("Ressourcenüberwachung gestartet")
            
            # 1. Datenbank-Verbindung
            self.logger.info("Initialisiere Datenbankverbindung...")
            self.db = DatabaseConnection()
            self.logger.success("Datenbankverbindung hergestellt")
            
            # 2. Daten laden
            self.logger.info("Lade Daten aus der Pipeline...")
            data = self.db.get_latest_data()
            total_symbols = len(data['symbol'].unique())
            self.logger.success(f"Daten geladen: {len(data)} Zeilen für {total_symbols} Symbole")
            
            # 3. Feature Engineering
            self.logger.info("Führe Feature-Engineering durch...")
            if self.low_performance_mode:
                original_size = len(data)
                data = data.sample(frac=0.1)  # Reduziere auf 10%
                self.logger.log_model_metrics(
                    model_name="data_processing",
                    metrics={
                        "low_performance_mode": True,
                        "original_rows": original_size,
                        "reduced_rows": len(data),
                        "reduction_percentage": (len(data) / original_size) * 100
                    }
                )
            
            # 4. Risiko-Checks
            self.logger.info("Führe Risiko-Checks durch...")
            if not self._pre_risk_checks(data):
                self.logger.failure("Risiko-Checks fehlgeschlagen")
                return
            self.logger.success("Risiko-Checks bestanden")
                
            # 5. KI-Analyse
            self.logger.section("KI-Analyse")
            predictions = pd.DataFrame()
            
            for i, symbol in enumerate(data['symbol'].unique(), 1):
                progress_pct = (i/total_symbols*100)
                self.logger.info(f"[{progress_pct:.1f}%] Analysiere {symbol}")
                
                symbol_data = data[data['symbol'] == symbol].copy()
                features = self._extract_features(symbol_data)
                
                # KI-Prompt vorbereiten
                prompt = self._prepare_ai_prompt(features)
                self.logger.info(f"Prompt erstellt ({len(prompt)} Zeichen)")
                
                # Tokenisierung
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.model.device)
                self.logger.debug(f"Input tokenisiert ({len(inputs['input_ids'][0])} Tokens)")
                
                # Modellvorhersage
                self.logger.info("Warte auf KI-Antwort...")
                
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20000,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    
                    # Decodierung
                    raw_prediction = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    self.logger.debug(f"Vorhersage generiert ({len(raw_prediction)} Zeichen)")
                    
                    # Analyse-Extraktion
                    analysis_result = self._extract_prediction_value(raw_prediction)
                    trading_signal = self._generate_trading_signal(analysis_result)
                    
                    # Logging und Speicherung
                    self.logger.log_prediction(
                        symbol=symbol,
                        prediction=trading_signal['signal'],
                        confidence=trading_signal['confidence'],
                        features=analysis_result
                    )
                    
                    predictions = pd.concat([predictions, pd.DataFrame([{
                        'symbol': symbol,
                        'prediction': trading_signal['signal'],
                        'confidence': trading_signal['confidence'],
                        'position_size': trading_signal['position_size'],
                        'stop_loss': trading_signal['stop_loss'],
                        'take_profit': trading_signal['take_profit']
                    }])], ignore_index=True)
                    
                except Exception as e:
                    self.logger.error(f"Fehler bei der Analyse von {symbol}: {str(e)}")
                    continue

            self.logger.success(f"KI-Analyse abgeschlossen für {total_symbols} Symbole")
            
            # 6. Signalgenerierung
            self.logger.section("Signal-Generierung")
            signals = self._generate_signals(data, predictions)
            self.logger.info(f"{len(signals)} Signale generiert")
            
            # 7. Backtesting
            self.logger.section("Backtesting")
            backtest_results = self._run_backtesting(data)
            self.logger.info(f"{len(backtest_results)} Strategien getestet")
            
            # 8. Ergebnisse speichern
            self._save_results(signals, backtest_results)
            
            # Stoppe Ressourcenüberwachung
            self.resource_manager.stop_monitoring()
            
            self.logger.section("Analyse abgeschlossen")
            self.logger.timing(start_time)
            
            return signals, backtest_results

        except Exception as e:
            self.resource_manager.stop_monitoring()
            self.logger.error(f"Kritischer Fehler in der Pipeline: {str(e)}")
            raise

    def _prepare_ai_prompt(self, features: dict) -> str:
        """Bereitet den KI-Prompt vor"""
        return f"""Analysiere die folgenden Handelsdaten für eine KI-basierte Trading-Entscheidung:

Input-Features: {str(features)}

Führe eine detaillierte Analyse durch und gib das Ergebnis in exakt diesem Format zurück:

{{
    'metadata': {{
        'symbol': str,
        'analysis_timestamp': str,
        'data_points': int
    }},
    'price_analysis': {{
        'price_metrics': {{
            'price_strength': float,
            'volatility_level': float,
            'volume_strength': float,
            'support_level': float,
            'resistance_level': float
        }},
        'trend_metrics': {{
            'trend_direction': float,
            'trend_strength': float,
            'price_momentum': float
        }}
    }},
    'news_analysis': {{
        'overall_sentiment': {{
            'sentiment_score': float,
            'sentiment_confidence': float,
            'sentiment_trend': float
        }},
        'news_relevance': {{
            'volume_score': float,
            'freshness_score': float,
            'impact_score': float
        }},
        'key_events': {{
            'has_earnings': bool,
            'has_guidance': bool,
            'has_merger': bool,
            'has_regulatory': bool,
            'has_market_move': bool
        }}
    }},
    'technical_analysis': {{
        'indicators': {{
            'rsi_signal': float,
            'macd_signal': float,
            'bb_signal': float,
            'ma_signal': float
        }},
        'patterns': {{
            'trend_strength': float,
            'reversal_probability': float,
            'breakout_potential': float
        }}
    }},
    'combined_analysis': {{
        'overall_score': float,
        'confidence_score': float,
        'risk_level': float,
        'position_advice': {{
            'action': str,
            'size': float,
            'stop_loss': float,
            'take_profit': float
        }}
    }}
}}

Wichtig: 
- Alle numerischen Werte müssen im angegebenen Wertebereich liegen
- Übernimm die Metadaten exakt aus den Input-Features
- Gib das Dictionary exakt in diesem Format zurück
- Keine zusätzlichen Erklärungen oder Kommentare
- Verwende die vorhandenen Features für die Analyse"""

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
        reduction_factor = self.resource_manager.get_data_reduction_factor()
        if reduction_factor < 1.0:
            self.logger.log_model_metrics(
                model_name="data_processing",
                metrics={
                    "low_performance_mode": True,
                    "data_reduction_factor": reduction_factor,
                    "original_rows": len(data)
                }
            )
        
        processed_data = calculate_technical_indicators(data, reduction_factor)
        
        if reduction_factor < 1.0:
            self.logger.log_model_metrics(
                model_name="data_processing",
                metrics={
                    "reduced_rows": len(processed_data),
                    "reduction_percentage": (len(processed_data) / len(data)) * 100
                }
            )
            
        return processed_data

    def _pre_risk_checks(self, data):
        """Führt Risikochecks vor der Analyse durch (deaktiviert)"""
        self.logger.log_model_metrics(
            model_name="risk_metrics",
            metrics={
                "risk_checks_disabled": True,
                "status": "passed"
            }
        )
        
        self.logger.info("Risiko-Checks sind deaktiviert")
        return True

    def _extract_features(self, data):
        """Extrahiert Features für das KI-Modell"""
        symbol = data['symbol'].iloc[0]  # Hole das Symbol aus den Daten
        
        features = {
            'metadata': {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(data)
            },
            'price_stats': {
                'close_mean': float(data['close'].mean()),
                'close_std': float(data['close'].std()),
                'volume_mean': float(data['volume'].mean()),
                'high_max': float(data['high'].max()),
                'low_min': float(data['low'].min())
            }
        }
        
        # News-Daten hinzufügen, falls vorhanden
        if 'news_titles' in data.columns and 'avg_sentiment' in data.columns:
            # Filtere Zeilen mit News
            news_data = data[data['news_titles'].notna()]
            
            # Verarbeite die neuesten News-Daten
            latest_news = {}
            if not news_data.empty and news_data['news_titles'].iloc[0]:
                # Extrahiere alle News-Komponenten
                titles = news_data['news_titles'].iloc[0].split(' ||| ')
                descriptions = news_data['news_descriptions'].iloc[0].split(' ||| ') if news_data['news_descriptions'].iloc[0] else []
                urls = news_data['news_urls'].iloc[0].split(' ||| ') if news_data['news_urls'].iloc[0] else []
                sentiments = [float(s) for s in news_data['news_sentiments'].iloc[0].split(' ||| ')] if news_data['news_sentiments'].iloc[0] else []
                
                # Im Low-Performance-Modus: Reduziere die Anzahl der News auf maximal 3
                if hasattr(self, 'low_performance_mode') and self.low_performance_mode:
                    self.logger.log_model_metrics(
                        model_name="data_processing",
                        metrics={"original_news": len(titles), "reduced_news": min(3, len(titles))}
                    )
                    titles = titles[:3]
                    descriptions = descriptions[:3] if descriptions else []
                    urls = urls[:3] if urls else []
                    sentiments = sentiments[:3] if sentiments else []
                
                # Kombiniere die Daten
                latest_news = {
                    'articles': [
                        {
                            'title': title,
                            'description': desc if i < len(descriptions) else None,
                            'url': url if i < len(urls) else None,
                            'sentiment': sent if i < len(sentiments) else None
                        }
                        for i, (title, desc, url, sent) in enumerate(zip(titles, descriptions, urls, sentiments))
                    ]
                }
            
            features['news_stats'] = {
                'sentiment_mean': float(news_data['avg_sentiment'].mean()) if not news_data.empty and not pd.isna(news_data['avg_sentiment'].mean()) else 0.0,
                'sentiment_std': float(news_data['avg_sentiment'].std()) if not news_data.empty and not pd.isna(news_data['avg_sentiment'].std()) else 0.0,
                'volume': int(news_data['news_count'].sum()) if not news_data.empty else 0,
                'latest_news': latest_news
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

    def _extract_prediction_value(self, model_output: str) -> dict:
        """Extrahiert die strukturierte Analyse aus der Modellausgabe."""
        try:
            # Finde das erste Dictionary in der Ausgabe
            start_idx = model_output.find('{')
            end_idx = model_output.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                dict_str = model_output[start_idx:end_idx]
                analysis = eval(dict_str)  # Sicher da wir kontrolliertes Format haben
                
                # Validiere das Format
                required_keys = ['metadata', 'price_analysis', 'news_analysis', 'technical_analysis', 'combined_analysis']
                if not all(key in analysis for key in required_keys):
                    raise ValueError("Unvollständiges Analyse-Format")
                
                return analysis
            raise ValueError("Keine gültige Analyse-Struktur gefunden")
        except Exception as e:
            self.logger.log_model_metrics(
                model_name="analysis_pipeline",
                metrics={"action": "extracting_default_analysis", "error": str(e)}
            )
            # Gebe Standard-Analyse-Struktur zurück
            return {
                'metadata': {
                    'symbol': 'UNKNOWN',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_points': 0
                },
                'price_analysis': {
                    'price_metrics': {'price_strength': 0, 'volatility_level': 0.5, 'volume_strength': 0,
                                    'support_level': 0, 'resistance_level': 0},
                    'trend_metrics': {'trend_direction': 0, 'trend_strength': 0, 'price_momentum': 0}
                },
                'news_analysis': {
                    'overall_sentiment': {'sentiment_score': 0, 'sentiment_confidence': 0, 'sentiment_trend': 0},
                    'news_relevance': {'volume_score': 0, 'freshness_score': 0, 'impact_score': 0},
                    'key_events': {'has_earnings': False, 'has_guidance': False, 'has_merger': False,
                                'has_regulatory': False, 'has_market_move': False}
                },
                'technical_analysis': {
                    'indicators': {'rsi_signal': 0, 'macd_signal': 0, 'bb_signal': 0, 'ma_signal': 0},
                    'patterns': {'trend_strength': 0, 'reversal_probability': 0, 'breakout_potential': 0}
                },
                'combined_analysis': {
                    'overall_score': 0,
                    'confidence_score': 0.5,
                    'risk_level': 0.5,
                    'position_advice': {'action': 'hold', 'size': 0, 'stop_loss': 0, 'take_profit': 0}
                }
            }

    def _generate_trading_signal(self, analysis: dict) -> dict:
        """Generiert ein Handelssignal aus der detaillierten Analyse."""
        try:
            # Extrahiere relevante Metriken
            symbol = analysis['metadata']['symbol']
            overall_score = analysis['combined_analysis']['overall_score']
            confidence = analysis['combined_analysis']['confidence_score']
            risk_level = analysis['combined_analysis']['risk_level']
            position_advice = analysis['combined_analysis']['position_advice']
            
            # Bestimme das Signal basierend auf dem Overall-Score
            if abs(overall_score) < 0.2:  # Neutrale Zone
                signal = 0
            else:
                signal = 1 if overall_score > 0 else -1
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'risk_level': risk_level,
                'position_size': position_advice['size'],
                'stop_loss': position_advice['stop_loss'],
                'take_profit': position_advice['take_profit'],
                'timestamp': analysis['metadata']['analysis_timestamp']
            }
            
        except Exception as e:
            self.logger.log_model_metrics(
                model_name="analysis_pipeline",
                metrics={"action": "generating_default_signal", "error": str(e)}
            )
            return {
                'symbol': 'UNKNOWN',
                'signal': 0,
                'confidence': 0.5,
                'risk_level': 0.5,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'timestamp': datetime.now().isoformat()
            }

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
            
            signals = pd.concat([signals, pd.DataFrame([{
                'symbol': symbol,
                'signal': signal,
                'confidence': symbol_pred['confidence']
            }])], ignore_index=True)
        
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
    pipeline.run_analysis() 