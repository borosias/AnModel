import {type HistoryItem} from '../types';

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
