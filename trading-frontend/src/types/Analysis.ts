export interface MarketAnalysis {
    id: number;
    timestamp: string;
    market_data: string;
    analysis_result: string;
    confidence_score: number;
    recommendations: string[];
}

export interface NewsAnalysis {
    id: number;
    timestamp: string;
    news_title: string;
    news_content: string;
    sentiment_score: number;
    impact_analysis: string;
}

export interface CombinedAnalysis {
    market_analysis: MarketAnalysis;
    news_analysis: NewsAnalysis;
    overall_recommendation: string;
    risk_level: 'low' | 'medium' | 'high';
}

export interface AIAnalysis {
    timestamp: string;
    analysis: string;
}
