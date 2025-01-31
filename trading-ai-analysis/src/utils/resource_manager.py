import psutil
import time
from threading import Thread
import torch
from typing import Optional
from .ai_logger import AILogger
from .console_logger import ConsoleLogger

class ResourceManager:
    def __init__(self, max_cpu_percent: float = 85.0, max_memory_percent: float = 85.0):
        """
        Initialisiert den ResourceManager mit Grenzwerten für CPU und RAM.
        
        Args:
            max_cpu_percent: Maximale CPU-Auslastung in Prozent (default: 85%)
            max_memory_percent: Maximale RAM-Auslastung in Prozent (default: 85%)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.logger = AILogger(name="resource_manager")
        self.console = ConsoleLogger(name="resource_manager")
        self._monitoring_thread: Optional[Thread] = None
        self._should_monitor = False
        self._low_performance_mode = not torch.cuda.is_available()
        
        if self._low_performance_mode:
            self.console.warning("GPU nicht verfügbar - Low-Performance-Modus aktiviert")
            self.logger.log_model_metrics(
                "resource_manager",
                {"mode": "low_performance", "reason": "no_gpu"}
            )
        
    def is_low_performance_mode(self) -> bool:
        """
        Gibt zurück, ob der Low-Performance-Modus aktiv ist.
        
        Returns:
            bool: True wenn im Low-Performance-Modus, sonst False
        """
        return self._low_performance_mode
        
    def get_data_reduction_factor(self) -> float:
        """
        Gibt den Faktor zurück, um den die Datenmenge reduziert werden soll.
        
        Returns:
            float: Reduktionsfaktor (1.0 = keine Reduktion, 0.1 = auf 10% reduzieren)
        """
        if self._low_performance_mode:
            return 0.1  # Reduziere auf 10% im Low-Performance-Modus
        return 1.0
        
    def start_monitoring(self):
        """Startet die Ressourcenüberwachung in einem separaten Thread."""
        if self._monitoring_thread is not None:
            return
            
        self._should_monitor = True
        self._monitoring_thread = Thread(target=self._monitor_resources, daemon=True)
        self._monitoring_thread.start()
        self.console.info("Ressourcenüberwachung gestartet")
        
    def stop_monitoring(self):
        """Stoppt die Ressourcenüberwachung."""
        self._should_monitor = False
        if self._monitoring_thread is not None:
            self._monitoring_thread.join()
            self._monitoring_thread = None
        self.console.info("Ressourcenüberwachung gestoppt")
        
    def _monitor_resources(self):
        """Überwacht kontinuierlich die Systemressourcen."""
        while self._should_monitor:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'cpu_limit': self.max_cpu_percent,
                'memory_limit': self.max_memory_percent
            }
            
            self.logger.log_model_metrics("resource_monitor", metrics)
            
            if cpu_percent > self.max_cpu_percent:
                self.console.warning(f"CPU-Auslastung zu hoch: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                self._throttle_resources()
                
            if memory_percent > self.max_memory_percent:
                self.console.warning(f"RAM-Auslastung zu hoch: {memory_percent:.1f}% > {self.max_memory_percent}%")
                self._throttle_resources()
                
            time.sleep(5)  # Überprüfung alle 5 Sekunden
            
    def _throttle_resources(self):
        """Reduziert die Ressourcennutzung bei Überschreitung der Grenzwerte."""
        # GPU-Cache leeren, falls verfügbar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Garbage Collection forcieren
        import gc
        gc.collect()
        
    def get_available_resources(self):
        """Gibt die aktuell verfügbaren Ressourcen zurück."""
        cpu_available = 100 - psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_available = 100 - memory.percent
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f'gpu_{i}'] = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # In GB
                    'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3     # In GB
                }
        
        return {
            'cpu_available': cpu_available,
            'memory_available': memory_available,
            'gpu_info': gpu_info
        } 