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

export type Features = Record<string, any>;

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