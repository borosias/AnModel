import { useState, useEffect } from 'react';
import type {HistoryItem} from '../types';

export const usePredictionHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem("predictionHistory");
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        const historyData = parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        }));
        setHistory(historyData.slice(0, 20));
      } catch (e) {
        console.error("Помилка завантаження історії:", e);
      }
    }
  }, []);

  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem("predictionHistory", JSON.stringify(history));
    }
  }, [history]);

  const addToHistory = (item: HistoryItem) => {
    setHistory(prev => [item, ...prev.slice(0, 19)]);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("predictionHistory");
  };

  return { history, addToHistory, clearHistory };
};