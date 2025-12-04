export interface ServiceDetail {
    status: string;
    features: string[];
    description?: string;
}

export interface ServicesResponse {
    services: string[];
    details: Record<string, ServiceDetail>;
}

export interface HealthResponse {
    status: string;
    timestamp?: string;
}

export interface Features {
    days_since_first?: number
    days_since_last?: number
    distinct_items?: number
    events_per_day?: number
    last_event_type?: string
    last_item?: string
    last_region?: string
    snapshot_date?: string
    total_clicks?: number
    total_events?: number
    total_purchases?: number
    total_spent?: number
    trend_popularity_max?: number
    trend_popularity_mean?: number
}

export interface User {
    user_id: string;
    features?: Features
    events_per_day?: number;
    total_events?: number;
    total_purchases?: number;
    distinct_items?: number;
    days_since_last?: number;
    avg_spend_per_purchase_30d?: number;
}

export interface PredictionResult {
    predictions: any[];
    model?: string;
    timestamp?: string;
}

export interface HistoryItem {
    id: string;
    timestamp: Date;
    service: string;
    input: Record<string, any>;
    output: any[];
    model: string;
    user_id?: string;
}

export interface ModelInfo {
    service: string;
    features: string[];
    optimal_threshold?: number;
    feature_importance_top10?: any[];
    feature_medians?: Record<string, number>;
}

export interface GraphTab {
    label: string;
    icon: React.ReactNode;
}

export type InputMode = 'manual' | 'db';