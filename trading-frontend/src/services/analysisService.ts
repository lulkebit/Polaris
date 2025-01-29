import { AIAnalysis } from '../types/Analysis';

const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

export const analysisService = {
    async getLatestAnalyses(): Promise<AIAnalysis[]> {
        const response = await fetch(`${API_BASE_URL}/analyses/latest`);
        if (!response.ok) {
            throw new Error('Fehler beim Laden der Analysen');
        }
        return response.json();
    },

    async getAnalysisByDate(date: string): Promise<AIAnalysis[]> {
        const response = await fetch(
            `${API_BASE_URL}/analyses/by-date/${date}`
        );
        if (!response.ok) {
            throw new Error(
                'Fehler beim Laden der Analysen für das angegebene Datum'
            );
        }
        return response.json();
    },

    async getMarketAnalysis(id: number): Promise<MarketAnalysis> {
        const response = await fetch(`${API_BASE_URL}/market-analysis/${id}`);
        if (!response.ok) {
            throw new Error('Fehler beim Laden der Marktanalyse');
        }
        return response.json();
    },

    async getNewsAnalysis(id: number): Promise<NewsAnalysis> {
        const response = await fetch(`${API_BASE_URL}/news-analysis/${id}`);
        if (!response.ok) {
            throw new Error('Fehler beim Laden der Nachrichtenanalyse');
        }
        return response.json();
    },
};
