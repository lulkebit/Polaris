from ..models.deepseek_model import DeepseekAnalyzer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketAnalyzer:
    def __init__(self):
        logger.info("Initialisiere MarketAnalyzer")
        self.model = DeepseekAnalyzer()

    def analyze_data(self, market_data=None, news_data=None):
        """
        Führt eine Analyse der bereitgestellten Daten durch
        """
        logger.info("Starte Datenanalyse")
        
        if market_data is None and news_data is None:
            logger.error("Keine Daten für die Analyse bereitgestellt")
            return None
            
        if market_data is not None and news_data is not None:
            logger.info("Führe kombinierte Analyse durch")
            return self.model.get_combined_analysis(market_data, news_data)
        elif market_data is not None:
            logger.info("Führe nur Marktdatenanalyse durch")
            return self.model.analyze_market_data(market_data)
        else:
            logger.info("Führe nur Nachrichtenanalyse durch")
            return self.model.analyze_news_data(news_data) 