export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const TRANSLATIONS: Record<string, string> = {
  total_events: "Кількість подій",
  total_purchases: "Кількість покупок",
  days_since_last_event: "Днів з останньої активності",
  purchase_proba: "Ймовірність покупки",
  will_purchase_pred: "Прогноз покупки",
  days_to_next_pred: "Дні до наступної покупки",
  next_purchase_amount_pred: "Сума наступної покупки",
  churn_probability: "Риск відтоку",
  lifetime_value: "Життєва цінність клієнта",
  engagement_score: "Рівень залучення",
  avg_spend_per_purchase_30d: "Середній чек (30 днів)",
  events_per_day: "Подій на день",
  distinct_items: "Унікальних товарів",
  total_spent: "Загальна сума витрат",
  loaded: "Активна",
  error: "Помилка",
  pending: "Очікування",
  predictions: "Прогнози",
  records: "Записи",
  model: "Модель",
  features: "Характеристики",
};

export const GRAPH_TABS = [
  { label: "Лінійний графік", icon: "TimelineIcon" },
  { label: "Стовпчикова діаграма", icon: "EqualizerIcon" },
  { label: "Обласний графік", icon: "AreaChart" },
  { label: "Кругова діаграма", icon: "PieChart" },
  { label: "Радарна діаграма", icon: "RadarChart" },
] as const;

export const DEFAULT_VALUES: Record<string, number> = {
  total_events: 100,
  total_purchases: 3,
  days_since_last_event: 5,
  avg_spend_per_purchase_30d: 1200,
};