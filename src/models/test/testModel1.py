# test_fix.py
import pandas as pd
import numpy as np
from models.models.context_aware import ContextAwareModel

# Создаем тестовые данные
test_data = {
    'user_id': ['user1', 'user2', 'user3'],
    'total_events': [100, 200, 150],
    'days_since_last': [5, 2, 10],
    'category': ['electronics', 'clothing', 'clothes'],  # Категориальная колонка
    'target_will_purchase': [1, 0, 1]
}

df = pd.DataFrame(test_data)

print("🧪 Тестируем исправление...")
print(f"Данные:\n{df}")
print(f"\nКолонки: {list(df.columns)}")
print(f"Типы колонок:\n{df.dtypes}")

# Тестируем модель
model = ContextAwareModel()

try:
    # Обучаем
    print("\n🚀 Обучаем модель...")
    model.train(df, epochs=5)
    print("✅ Обучение успешно")

    # Предсказываем
    print("\n🔮 Делаем предсказания...")
    predictions = model.predict(df.head(2))
    print(f"✅ Предсказания: {predictions}")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()