import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import colorama
from colorama import Fore, Style
import json
import traceback

class ConsoleLogger:
    """
    Basis-Logger für die Konsolen-Ausgabe mit farbiger Formatierung.
    Verhindert doppeltes Logging durch Singleton-Pattern.
    """
    
    _instance = None
    
    # Farbdefinitionen für verschiedene Log-Level und Kategorien
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'PROCESS': Fore.BLUE,
        'DATA': Fore.MAGENTA,
        'METRIC': Fore.CYAN,
        'SYSTEM': Fore.WHITE
    }
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConsoleLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "trading_console",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        debug_file: Optional[str] = None
    ):
        """
        Initialisiert den ConsoleLogger als Singleton.
        
        Args:
            name: Name des Loggers
            log_level: Logging-Level (DEBUG, INFO, etc.)
            log_file: Optional - Pfad zur Log-Datei
            debug_file: Optional - Pfad zur Debug-Log-Datei
        """
        if hasattr(self, 'initialized'):
            return
            
        colorama.init()
        
        # Root Logger konfigurieren
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)  # Immer DEBUG für maximale Details
        
        # Alle bestehenden Handler entfernen
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Formatter für verschiedene Ausgaben
        self.console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.debug_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler (mit Level-Filter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        console_handler.setLevel(log_level)
        console_handler.addFilter(self.ColoredOutputFilter())
        self.logger.addHandler(console_handler)
        
        # File Handler für normales Logging
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(self.file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
        
        # Separater Debug File Handler
        if debug_file:
            debug_path = Path(debug_file)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_handler = logging.FileHandler(debug_file, encoding='utf-8')
            debug_handler.setFormatter(self.debug_formatter)
            debug_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(debug_handler)
        
        self.start_time = datetime.now()
        self.initialized = True

    class ColoredOutputFilter(logging.Filter):
        def filter(self, record):
            if record.levelname in ConsoleLogger.COLORS:
                color = ConsoleLogger.COLORS[record.levelname]
                record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
            return True
    
    def _format_dict(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Formatiert ein Dictionary für übersichtliche Ausgabe"""
        output = []
        for key, value in data.items():
            if isinstance(value, dict):
                output.append(f"{'  ' * indent}{key}:")
                output.append(self._format_dict(value, indent + 1))
            elif isinstance(value, (list, tuple)):
                output.append(f"{'  ' * indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        output.append(self._format_dict(item, indent + 1))
                    else:
                        output.append(f"{'  ' * (indent + 1)}- {item}")
            else:
                output.append(f"{'  ' * indent}{key}: {value}")
        return "\n".join(output)
    
    def debug(self, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Erweitertes Debug-Level Logging mit Extra-Informationen"""
        if extra:
            msg = f"{msg}\nDetails:\n{self._format_dict(extra)}"
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, category: str = None, **kwargs):
        """Info-Level Logging mit optionaler Kategorie"""
        if category and category in self.COLORS:
            msg = f"{self.COLORS[category]}{msg}{Style.RESET_ALL}"
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, details: Optional[Dict[str, Any]] = None, **kwargs):
        """Warning-Level Logging mit optionalen Details"""
        if details:
            msg = f"{msg}\nWarnung-Details:\n{self._format_dict(details)}"
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, exc_info: bool = True, **kwargs):
        """Error-Level Logging mit Stack Trace"""
        if exc_info:
            msg = f"{msg}\n{traceback.format_exc()}"
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, system_info: bool = True, **kwargs):
        """Critical-Level Logging mit System-Informationen"""
        if system_info:
            import psutil
            system_details = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
            msg = f"{msg}\nSystem-Status:\n{self._format_dict(system_details)}"
        self.logger.critical(msg, *args, **kwargs)
    
    def section(self, title: str, width: int = 80, category: str = None):
        """Erstellt einen hervorgehobenen Abschnitt mit optionaler Kategorie"""
        separator = "=" * width
        if category and category in self.COLORS:
            color = self.COLORS[category]
            centered_title = f"{color}{title.center(width)}{Style.RESET_ALL}"
        else:
            centered_title = title.center(width)
        
        self.info(separator)
        self.info(centered_title)
        self.info(separator)
        
        # Debug-Information über Sektionsstart
        self.debug(f"Sektion gestartet: {title}", extra={
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'elapsed_time': str(datetime.now() - self.start_time)
        })
    
    def success(self, msg: str, *args, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Erfolgs-Nachricht mit optionalen Metriken"""
        if metrics:
            msg = f"✓ {msg}\nMetriken:\n{self._format_dict(metrics)}"
        else:
            msg = f"✓ {msg}"
        self.info(msg, *args, **kwargs)
    
    def failure(self, msg: str, *args, error_details: Optional[Dict[str, Any]] = None, **kwargs):
        """Fehler-Nachricht mit detaillierten Fehlerinformationen"""
        if error_details:
            msg = f"✗ {msg}\nFehler-Details:\n{self._format_dict(error_details)}"
        else:
            msg = f"✗ {msg}"
        self.error(msg, *args, **kwargs)
    
    def timing(self, start_time: datetime, operation: str = None):
        """Erweiterte Zeitmessung mit Operation-Details"""
        elapsed = datetime.now() - start_time
        details = {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': elapsed.total_seconds(),
            'operation': operation or 'Unspezifiziert'
        }
        self.info(f"Ausführungszeit: {elapsed}", category='METRIC', extra=details)
    
    def data_info(self, msg: str, data: Dict[str, Any]):
        """Spezielles Logging für Dateninformationen"""
        formatted_data = self._format_dict(data)
        self.info(f"{msg}\n{formatted_data}", category='DATA')
    
    def process_info(self, msg: str, process_details: Dict[str, Any]):
        """Spezielles Logging für Prozessinformationen"""
        formatted_details = self._format_dict(process_details)
        self.info(f"{msg}\n{formatted_details}", category='PROCESS')
    
    def system_info(self, msg: str, system_details: Dict[str, Any]):
        """Spezielles Logging für Systeminformationen"""
        formatted_details = self._format_dict(system_details)
        self.info(f"{msg}\n{formatted_details}", category='SYSTEM')

    def ai_prompt(self, prompt: str, model: str = "Unspezifiziert"):
        """
        Loggt einen KI-Prompt mit spezieller Formatierung.
        
        Args:
            prompt: Der an die KI gesendete Prompt
            model: Das verwendete KI-Modell
        """
        self.section(f"KI-Prompt ({model})")
        formatted_prompt = f"{Fore.BLUE}{prompt}{Style.RESET_ALL}"
        self.info(formatted_prompt)
        self._last_prompt_time = datetime.now()
        self._last_prompt_model = model
        self._last_prompt_length = len(prompt)
        
        # Geschätzte Wartezeit basierend auf Prompt-Länge
        # Durchschnittliche Verarbeitungszeit: ~100 Zeichen/Sekunde
        self._estimated_wait_time = max(5, self._last_prompt_length / 100)
        self.info(f"Geschätzte Wartezeit: {self._estimated_wait_time:.1f} Sekunden", category='METRIC')
        
    def ai_response(self, response: str):
        """
        Loggt die Antwort der KI mit spezieller Formatierung.
        
        Args:
            response: Die Antwort der KI
        """
        if hasattr(self, '_last_prompt_time'):
            elapsed_time = datetime.now() - self._last_prompt_time
            self.info(f"Antwortzeit: {elapsed_time.total_seconds():.2f} Sekunden", category='METRIC')
        
        self.section("KI-Antwort")
        formatted_response = f"{Fore.MAGENTA}{response}{Style.RESET_ALL}"
        self.info(formatted_response)
        
    def ai_waiting(self):
        """
        Zeigt den Wartestatus für die KI-Antwort an und aktualisiert die Wartezeit.
        """
        if not hasattr(self, '_last_prompt_time'):
            self._last_prompt_time = datetime.now()
            self._last_prompt_model = "Unspezifiziert"
            self._estimated_wait_time = 10  # Standardwert
        
        elapsed_time = datetime.now() - self._last_prompt_time
        remaining_time = max(0, self._estimated_wait_time - elapsed_time.total_seconds())
        
        progress = min(100, (elapsed_time.total_seconds() / self._estimated_wait_time) * 100)
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(elapsed_time.total_seconds()) % 10]
        
        # Status-Nachricht mit Fortschritt
        status_msg = (
            f"\r{spinner} Warte auf KI-Antwort... "
            f"({elapsed_time.total_seconds():.1f}s / ~{self._estimated_wait_time:.1f}s) "
            f"[{progress:.0f}%]"
        )
        
        if remaining_time > 0:
            status_msg += f" - Noch ~{remaining_time:.1f}s"
        else:
            status_msg += " - Antwort wird jeden Moment erwartet"
        
        # Direkte Konsolenausgabe mit Carriage Return
        print(f"{Fore.BLUE}{status_msg}{Style.RESET_ALL}", end="", flush=True)
        
        # Debug-Logging für detaillierte Informationen
        self.debug("KI-Verarbeitung läuft", extra={
            'wartezeit_sekunden': elapsed_time.total_seconds(),
            'geschätzte_restzeit': remaining_time,
            'prompt_länge': getattr(self, '_last_prompt_length', 0),
            'model': self._last_prompt_model,
            'start_zeit': self._last_prompt_time.strftime(self.TIME_FORMAT) if hasattr(self, 'TIME_FORMAT') else self._last_prompt_time.strftime('%H:%M:%S')
        })
        
    def ai_thinking(self, partial_response: str):
        """
        Zeigt Zwischenergebnisse während der KI-Generierung an.
        
        Args:
            partial_response: Die bisherige Teil-Antwort der KI
        """
        self.info(f"{Fore.CYAN}KI denkt: {partial_response}{Style.RESET_ALL}") 