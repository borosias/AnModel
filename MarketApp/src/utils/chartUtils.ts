import {type HistoryItem} from '../types';
import {TRANSLATIONS} from '../constants';

export const getPredictionData = (history: HistoryItem[]) => {
    return history.map((item, index) => {
        const output = item.output[0] || {};
        return {
            id: index + 1,
            time: item.timestamp.toLocaleTimeString("uk-UA", {
                hour: '2-digit',
                minute: '2-digit'
            }),
            date: item.timestamp.toLocaleDateString('uk-UA'),
            ...output,
            ...Object.fromEntries(
                Object.entries(item.input).filter(([key]) =>
                    typeof item.input[key] === 'number'
                )
            ),
        };
    });
};

export const getMetricKeys = (history: HistoryItem[]) => {
    if (history.length === 0) return [];

    const keys = new Set<string>();
    history.forEach(item => {
        const output = item.output[0] || {};
        Object.keys(output).forEach(key => {
            if (typeof output[key] === 'number') {
                keys.add(key);
            }
        });
    });

    return Array.from(keys);
};

export const getPieChartData = (history: HistoryItem[], chartColors: string[]) => {
    if (history.length === 0) return [];

    const lastPrediction = history[0]?.output[0] || {};
    return Object.entries(lastPrediction)
        .filter(([_, value]) => typeof value === 'number' && value > 0)
        .slice(0, 6)
        .map(([key, value], index) => ({
            name: TRANSLATIONS[key] || key.replace(/_/g, " "),
            value: value as number,
            fill: chartColors[index % chartColors.length],
        }));
};

export const getRadarChartData = (history: HistoryItem[]) => {
    if (history.length === 0) return [];

    const lastPrediction = history[0]?.output[0] || {};
    return Object.entries(lastPrediction)
        .filter(([_, value]) => typeof value === 'number')
        .slice(0, 8)
        .map(([key, value]) => ({
            subject: TRANSLATIONS[key]?.length > 15
                ? TRANSLATIONS[key].substring(0, 15) + "..."
                : TRANSLATIONS[key] || key,
            value: value as number,
            fullMark: Math.max(value as number * 1.5, 1),
        }));
};