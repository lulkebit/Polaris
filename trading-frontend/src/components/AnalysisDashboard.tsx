import { useState, useEffect } from 'react';
import { AIAnalysis } from '../types/Analysis';
import { analysisService } from '../services/analysisService';

const parseJsonFromText = (text: string, marker: string) => {
    try {
        const regex = new RegExp(`${marker}:[\\s\\S]*?({[\\s\\S]*?})(?=\\n\\s*\\n|$)`);
        const match = text.match(regex);
        if (match && match[1]) {
            return JSON.parse(match[1]);
        }
    } catch (e) {
        console.error(`Fehler beim Parsen von ${marker}:`, e);
    }
    return null;
};

const renderMarketData = (marketData: any) => {
    if (!marketData) return null;

    const symbols = Object.values(marketData.symbol);
    const dates = Object.values(marketData.date);
    const opens = Object.values(marketData.open);
    const closes = Object.values(marketData.close);
    const volumes = Object.values(marketData.volume);

    return (
        <div className="mt-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Marktdaten:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {symbols.map((symbol: string, index: number) => {
                    const change = ((closes[index] as number - opens[index] as number) / opens[index] as number) * 100;
                    return (
                        <div key={symbol} className="bg-gray-50 rounded-lg p-4 border">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-semibold">{symbol}</span>
                                <span className={change >= 0 ? 'text-green-600' : 'text-red-600'}>
                                    {change.toFixed(2)}%
                                </span>
                            </div>
                            <div className="text-sm space-y-1 text-gray-600">
                                <div>Eröffnung: ${opens[index].toFixed(2)}</div>
                                <div>Schluss: ${closes[index].toFixed(2)}</div>
                                <div>Volumen: {volumes[index].toLocaleString('de-DE')}</div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const renderNewsData = (newsData: any) => {
    if (!newsData) return null;

    const titles = Object.values(newsData.title);
    const contents = Object.values(newsData.content);
    const publishedDates = Object.values(newsData.published_at);

    return (
        <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Nachrichten:</h3>
            <div className="space-y-4">
                {titles.map((title: string, index: number) => (
                    <div key={index} className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="font-semibold text-gray-800 mb-2">{title}</h4>
                        <p className="text-sm text-gray-600 mb-2">{contents[index]}</p>
                        <time className="text-xs text-gray-500">
                            {new Date(publishedDates[index]).toLocaleString('de-DE')}
                        </time>
                    </div>
                ))}
            </div>
        </div>
    );
};

const renderAnalysis = (analysis: string) => {
    // Parse die JSON-Daten aus dem Text
    const marketData = parseJsonFromText(analysis, "Marktanalyse");
    const newsData = parseJsonFromText(analysis, "Nachrichtenanalyse");

    // Extrahiere den Einleitungstext
    const introText = analysis.split('\n')[0];

    return (
        <div className="text-gray-900">
            <div className="mb-4 font-medium">{introText}</div>
            {marketData && renderMarketData(marketData)}
            {newsData && renderNewsData(newsData)}
            <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Rohdaten:</h3>
                <div className="whitespace-pre-wrap font-mono text-sm overflow-x-auto">
                    {analysis.split('\n').map((line, index) => (
                        <div key={index} className="py-1">
                            {line}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export const AnalysisDashboard = () => {
    const [analyses, setAnalyses] = useState<AIAnalysis[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedDate, setSelectedDate] = useState(
        new Date().toISOString().split('T')[0]
    );

    useEffect(() => {
        loadAnalyses();
    }, [selectedDate]);

    const loadAnalyses = async () => {
        try {
            const data = await analysisService.getAnalysisByDate(selectedDate);
            setAnalyses(data);
            setError(null);
        } catch (err) {
            setError('Fehler beim Laden der Analysen');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className='flex items-center justify-center min-h-screen'>
                <div className='animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500'></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className='p-4 bg-red-100 border border-red-400 text-red-700 rounded'>
                {error}
            </div>
        );
    }

    return (
        <div className='w-full max-w-[95%] mx-auto px-4 py-8'>
            <div className='bg-white shadow rounded-lg p-6 mb-8'>
                <div className='flex justify-between items-center mb-6'>
                    <h2 className='text-xl font-semibold text-gray-800'>
                        Trading AI Analysen
                    </h2>
                    <div className='flex gap-4'>
                        <input
                            type='date'
                            value={selectedDate}
                            onChange={(e) => setSelectedDate(e.target.value)}
                            className='px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
                        />
                    </div>
                </div>

                {analyses.length === 0 ? (
                    <div className='text-center py-8 text-gray-500'>
                        Keine Analysen verfügbar für das ausgewählte Datum.
                    </div>
                ) : (
                    <div className='space-y-8'>
                        {analyses.map((analysis) => (
                            <div
                                key={analysis.timestamp}
                                className='border rounded-lg p-6 hover:bg-gray-50 transition-colors'
                            >
                                <div className='flex justify-between items-start mb-4'>
                                    <time className='text-sm text-gray-500'>
                                        {new Date(
                                            analysis.timestamp
                                        ).toLocaleString('de-DE', {
                                            dateStyle: 'full',
                                            timeStyle: 'short',
                                        })}
                                    </time>
                                </div>
                                {renderAnalysis(analysis.analysis)}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
