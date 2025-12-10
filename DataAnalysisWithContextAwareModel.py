import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from src.models.models.context_aware import ContextAwareModel
import shap

# Настройка отображения
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", '{:.2f}'.format)
plt.style.use('seaborn-v0_8')

# Загрузка модели
model = ContextAwareModel.load("./src/models/production_models/context_aware_model1.pkl")

# Загрузка данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Используем последнюю доступную папку со снапшотами (автопоиск)
SNAPSHOT_BASE = os.path.join(BASE_DIR, "src", "analytics", "data", "daily_features")
snapshots = sorted([d for d in os.listdir(SNAPSHOT_BASE) if d.startswith("snapshot_")])
if not snapshots:
    raise FileNotFoundError("Нет снапшотов в data/daily_features")
SNAPSHOT_DIR = os.path.join(SNAPSHOT_BASE, snapshots[-1])

print(f"📂 Загрузка данных из: {SNAPSHOT_DIR}")


def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(SNAPSHOT_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


df = load_dataset("daily_snapshot1")
print(f"Загружено {len(df)} записей")

# ==================== 1. ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ ====================

result = model.predict(df)

# Фичи, которые мы используем на фронтенде для UI
UI_FEATURES = [
    'events_last_7d',
    'days_since_last',
    'purchase_frequency',
    'avg_spend_per_event',
    'conversion_rate_30d',
    'total_purchases'
]

# Объединяем исходные данные и предсказания
out = pd.concat([df, result], axis=1)

print("\n" + "=" * 80)
print("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
print("=" * 80)
print(f"Порог модели (threshold): {model.optimal_threshold_:.3f}")
print(f"Средняя вероятность (avg_proba): {out['purchase_proba'].mean():.3f}")
print(f"Will Purchase = 1: {out['will_purchase_pred'].sum()} ({out['will_purchase_pred'].mean():.1%})")

# ==================== 2. АНАЛИЗ ПОРОГОВ ДЛЯ UI ====================

print("\n" + "=" * 80)
print("АНАЛИЗ ПОРОГОВЫХ ЗНАЧЕНИЙ (ДЛЯ ФРОНТЕНДА)")
print("=" * 80)

# Сегментируем пользователей по уверенности модели
# High: > 70%, Medium: 30-70%, Low: < 30%
out['segment'] = pd.cut(
    out['purchase_proba'],
    bins=[-0.1, 0.3, 0.7, 1.1],
    labels=['Low (Red)', 'Medium (Blue)', 'High (Green)']
)

print("\nСредние значения фич для каждого сегмента вероятности:")
print("-" * 60)
segment_stats = out.groupby('segment',observed=True)[UI_FEATURES].median()
print(segment_stats)

print("\n" + "=" * 80)
print("ДЕТАЛЬНЫЕ ДИАПАЗОНЫ (Квантили 25% - 75%)")
print("Используйте эти числа для настройки цветов на фронте")
print("=" * 80)

for feature in UI_FEATURES:
    print(f"\n🔹 {feature}:")

    # Берем "Зеленый" сегмент (высокая вероятность)
    high_segment = out[out['purchase_proba'] >= 0.6][feature]
    # Берем "Красный" сегмент (низкая вероятность)
    low_segment = out[out['purchase_proba'] <= 0.1][feature]

    if len(high_segment) > 0:
        q25_high = high_segment.quantile(0.25)
        median_high = high_segment.median()
        print(f"   Для ТОП-клиентов обычно: > {q25_high:.1f} (медиана {median_high:.1f})")

    if len(low_segment) > 0:
        q75_low = low_segment.quantile(0.75)
        median_low = low_segment.median()
        print(f"   Для АУТСАЙДЕРОВ обычно: < {q75_low:.1f} (медиана {median_low:.1f})")

    # Рекомендация порога
    if len(high_segment) > 0 and len(low_segment) > 0:
        # Простое правило: среднее между "плохим" максимумом и "хорошим" минимумом
        suggested_threshold = (q75_low + q25_high) / 2
        print(f"   👉 РЕКОМЕНДУЕМЫЙ ПОРОГ (Good): {suggested_threshold:.1f}")

# ==================== 3. SURROGATE TREE (Понятные правила) ====================

print("\n" + "=" * 80)
print("ГЕНЕРАЦИЯ ЧЕЛОВЕКОЧИТАЕМЫХ ПРАВИЛ (SURROGATE TREE)")
print("Модель сложная, но мы упростим её логику до 3-х главных условий")
print("=" * 80)

# Обучаем маленькое дерево решений, чтобы понять логику "Большого брата"
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
X_tree = df[UI_FEATURES].fillna(0)
y_tree = out['will_purchase_pred']  # Пытаемся предсказать решение основной модели

tree.fit(X_tree, y_tree)

rules = export_text(tree, feature_names=UI_FEATURES)
print(rules)

# Вытаскиваем важность фич для этого простого дерева
tree_importance = pd.DataFrame({
    'feature': UI_FEATURES,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТОП-3 Фичи, которые реально делят людей на Да/Нет (по суррогату):")
print(tree_importance.head(3))

# ==================== 4. ГОТОВЫЙ КОНФИГ ДЛЯ ФРОНТА ====================

print("\n" + "=" * 80)
print("🚀 ГОТОВЫЙ JSON-КОНФИГ ДЛЯ UserSearch.tsx")
print("Скопируйте эти значения в функцию classifyUserForUI")
print("=" * 80)


# Вычисляем финальные пороги для конфига
def get_safe_threshold(series_high, series_low, default):
    if len(series_high) == 0 or len(series_low) == 0:
        return default
    # Порог "Хорошо" = нижняя граница топ-25% лучших
    good = series_high.quantile(0.25)
    # Порог "Плохо" = верхняя граница топ-75% худших
    bad = series_low.quantile(0.75)
    return good, bad


high_mask = out['purchase_proba'] >= 0.6
low_mask = out['purchase_proba'] <= 0.1

ev7_good, ev7_bad = get_safe_threshold(out[high_mask]['events_last_7d'], out[low_mask]['events_last_7d'], (10, 0))
recency_good, recency_bad = get_safe_threshold(out[high_mask]['days_since_last'], out[low_mask]['days_since_last'],
                                               (7, 30))
freq_good, freq_bad = get_safe_threshold(out[high_mask]['purchase_frequency'], out[low_mask]['purchase_frequency'],
                                         (0.1, 0.01))
money_good, money_bad = get_safe_threshold(out[high_mask]['avg_spend_per_event'], out[low_mask]['avg_spend_per_event'],
                                           (150, 50))

print("const THRESHOLDS = {")
print(f"  // Активность за 7 дней (чем больше, тем лучше)")
print(f"  events7: {{ good: {int(ev7_good)}, bad: {int(ev7_bad)} }},")
print(f"  // Дней с последнего визита (чем меньше, тем лучше)")
print(f"  recency: {{ good: {int(recency_good)}, bad: {int(recency_bad)} }},")
print(f"  // Частота покупок (чем больше, тем лучше)")
print(f"  freq: {{ good: {freq_good:.2f}, bad: {freq_bad:.2f} }},")
print(f"  // Средний чек на событие (чем больше, тем лучше)")
print(f"  avgSpend: {{ good: {int(money_good)}, bad: {int(money_bad)} }}")
print("};")

print("\n" + "=" * 80)
print("SHAP SUMMARY (для подтверждения)")
print("=" * 80)

# Базовый SHAP (как раньше) для подтверждения
try:
    import lightgbm as lgb

    if hasattr(model, 'clf') and isinstance(model.clf, lgb.LGBMClassifier):
        explainer = shap.TreeExplainer(model.clf)
        # Берем сэмпл для скорости
        sample_df = model._prepare_features_infer(df).iloc[:500]
        shap_values = explainer.shap_values(sample_df)
        if isinstance(shap_values, list): shap_values = shap_values[1]

        shap_sum = pd.DataFrame({
            'feature': sample_df.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False).head(5)
        print("Топ-5 фич по SHAP (реальная модель):")
        print(shap_sum)
except Exception as e:
    print(f"SHAP пропущен: {e}")

print("\nАНАЛИЗ ЗАВЕРШЕН. Используйте данные из секции 4 для обновления фронтенда.")