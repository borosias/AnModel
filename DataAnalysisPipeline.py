import pandas as pd
from src.models.models.context_aware import ContextAwareModel

# 1. Загружаем модель
model = ContextAwareModel.load("src/models/production_models/context_aware_model1.pkl")

# 2. Загружаем train данные
train = pd.read_parquet("src/analytics/data/snapshots/model1/train.parquet")

print("=" * 60)
print("АНАЛИЗ TRAIN ДАННЫХ")
print("=" * 60)

# 3. Общая статистика
print(f"\nВсего строк: {len(train)}")
pos = train[train['will_purchase_next_7d'] == 1]
neg = train[train['will_purchase_next_7d'] == 0]
print(f"Положительных (will_purchase=1): {len(pos)} ({len(pos)/len(train):.2%})")
print(f"Отрицательных (will_purchase=0): {len(neg)} ({len(neg)/len(train):.2%})")

# 4. Сравнение фичей между pos и neg
print("\n--- СРАВНЕНИЕ ФИЧЕЙ ---")
for col in ['events_last_7d', 'events_last_30d', 'recency_score', 'conversion_rate_30d', 'days_since_last']:
    pos_mean = pos[col].mean()
    neg_mean = neg[col].mean()
    print(f"{col:25s}: pos={pos_mean:10.3f}, neg={neg_mean:10.3f}, diff={pos_mean - neg_mean:+.3f}")

# 5. Ищем АКТИВНЫХ юзеров с will_purchase=1
print("\n--- АКТИВНЫЕ ЮЗЕРЫ (events_last_7d > 100) ---")
active = train[train['events_last_7d'] > 100]
print(f"Всего активных: {len(active)}")
active_pos = active[active['will_purchase_next_7d'] == 1]
print(f"Из них will_purchase=1: {len(active_pos)} ({len(active_pos)/max(1,len(active)):.2%})")

# 6. Ищем юзеров с recency_score > 0.2 и will_purchase=1
print("\n--- СВЕЖИЕ ЮЗЕРЫ (recency_score > 0.2) ---")
fresh = train[train['recency_score'] > 0.2]
print(f"Всего свежих: {len(fresh)}")
fresh_pos = fresh[fresh['will_purchase_next_7d'] == 1]
print(f"Из них will_purchase=1: {len(fresh_pos)} ({len(fresh_pos)/max(1,len(fresh)):.2%})")

# 7. Твой конкретный случай — ищем похожие примеры
print("\n--- ПОХОЖИЕ НА ТВОЙ INPUT (events_last_7d > 400, recency_score ~ 0.25) ---")
similar = train[(train['events_last_7d'] > 400) & (train['recency_score'] > 0.2) & (train['recency_score'] < 0.4)]
print(f"Найдено похожих: {len(similar)}")
if len(similar) > 0:
    print(f"Из них will_purchase=1: {similar['will_purchase_next_7d'].sum()} ({similar['will_purchase_next_7d'].mean():.2%})")
    print(f"\nПример похожего (первый):")
    print(similar[['events_last_7d', 'recency_score', 'conversion_rate_30d', 'will_purchase_next_7d']].head(3))

# 8. Проверим даты
print("\n--- ДАТЫ ---")
print(f"Min snapshot_date: {train['snapshot_date'].min()}")
print(f"Max snapshot_date: {train['snapshot_date'].max()}")

# 9. Проверим: есть ли вообще активные юзеры с will_purchase=1?
print("\n--- КРИТИЧЕСКАЯ ПРОВЕРКА ---")
active_and_pos = train[(train['events_last_7d'] > 50) & (train['will_purchase_next_7d'] == 1)]
print(f"Активные (events_last_7d > 50) И купят: {len(active_and_pos)}")
if len(active_and_pos) > 0:
    print(f"Их средний recency_score: {active_and_pos['recency_score'].mean():.3f}")
    print(f"Их средний days_since_last: {active_and_pos['days_since_last'].mean():.1f}")
