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

export const formatFeatureValue = (key: string, value: number) => {
    switch (key) {
        case TRANSLATIONS.purchase_proba:
        case "purchase_proba":
            return `${(value * 100).toFixed(3)}%`;

        case TRANSLATIONS.days_to_next_pred:
        case "days_to_next_pred":
            return `${(value).toFixed(1)}`;

        case TRANSLATIONS.next_purchase_amount_pred:
        case "next_purchase_amount_pred":
            return `${(value).toFixed(2)}₴`;

        case TRANSLATIONS.will_purchase_pred:
        case "will_purchase_pred":
            return value === 1 ? "Буде" : "Не Буде";

        default:
            return value;
    }
};
