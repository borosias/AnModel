import type {HealthResponse, ServicesResponse, PredictionResult, User, ModelInfo} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const apiClient = {
  health: async (): Promise<HealthResponse> => {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  services: async (): Promise<ServicesResponse> => {
    const response = await fetch(`${API_BASE}/services`);
    if (!response.ok) throw new Error("Не вдалося отримати список сервісів");
    return response.json();
  },

  modelInfo: async (serviceName: string): Promise<ModelInfo> => {
    const response = await fetch(`${API_BASE}/model-info/${serviceName}`);
    if (!response.ok) throw new Error("Не вдалося завантажити інформацію про модель");
    return response.json();
  },

  getUsers: async (): Promise<{ users: User[] }> => {
    const response = await fetch(
      `${API_BASE}/users`
    );
    if (!response.ok) throw new Error("Не вдалося завантажити користувачів");
    return response.json();
  },

  predict: async (service: string, records: Record<string, any>[]): Promise<PredictionResult> => {
    const url = `${API_BASE}/predict?service=${service}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records, service }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail ||
        errorData.error ||
        `Помилка сервера: ${response.status}`
      );
    }

    return response.json();
  },
};