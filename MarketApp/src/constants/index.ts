export const TRANSLATIONS: Record<string, string> = {
    total_events: "Event Count",
    total_purchases: "Purchase Count",
    days_since_last_event: "Days Since Last Activity",
    purchase_proba: "Purchase Probability",
    will_purchase_pred: "Purchase Prediction",
    days_to_next_pred: "Days to Next Purchase",
    next_purchase_amount_pred: "Next Purchase Amount",
    churn_probability: "Churn Probability",
    lifetime_value: "Customer Lifetime Value",
    engagement_score: "Engagement Level",
    avg_spend_per_purchase_30d: "Average Check (30 days)",
    events_per_day: "Events per Day",
    events_last_7d: "Activity (7 days)",
    events_last_30d: "Activity (30 days)",
    purchases_last_30d: "Purchases (30 days)",
    spent_last_30d: "Spending (30 days)",
    distinct_items: "Unique Items",
    total_spent: "Total Spending",
    loaded: "Active",
    error: "Error",
    pending: "Pending",
    predictions: "Predictions",
    records: "Records",
    model: "Model",
    features: "Features",
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
        case "total_spent":
        case "spent_last_30d":
            return `${(value).toFixed(2)}₴`;

        case TRANSLATIONS.will_purchase_pred:
        case "will_purchase_pred":
            return value === 1 ? "Will Purchase" : "Will Not Purchase";

        default:
            if (Number.isInteger(value)) return value.toString();
            return value.toFixed(2);

    }
};
