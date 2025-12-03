import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.context_aware import ContextAwareModel
from models.cross_region import CrossRegionModel


class ModelPipeline:
    def __init__(self, data_dir: str = "../analytics/data/innovative_snapshots"):
        self.data_dir = Path(data_dir)
        self.models = {
            'model1': ContextAwareModel(),
            'model2': CrossRegionModel(),
        }
        self.data = {}

    def load_snapshots(self):
        """Загрузка снапшотов"""
        print("📂 Загружаю снапшоты...")

        for model_name in ['model1', 'model2']:
            model_path = self.data_dir / model_name

            if model_path.exists():
                self.data[model_name] = {}

                for dataset in ['train', 'val', 'test']:
                    file_path = model_path / f"{dataset}.parquet"

                    if file_path.exists():
                        try:
                            df = pd.read_parquet(file_path)
                            self.data[model_name][dataset] = df
                            print(f"  {model_name}/{dataset}: {len(df):,} строк")
                        except Exception as e:
                            print(f"  ❌ Ошибка загрузки {file_path}: {e}")
                    else:
                        print(f"  ⚠️ Файл не найден: {file_path}")
            else:
                print(f"  ⚠️ Папка не найдена: {model_path}")

    def train_all(self):
        """Обучение всех моделей"""
        print("\n🚀 Обучаю модели...")

        # Model 1
        if 'model1' in self.data:
            print("\n1️⃣ Model 1: Context-Aware")
            train_df = self.data['model1'].get('train')
            val_df = self.data['model1'].get('val')

            if train_df is not None and len(train_df) > 100:
                self.models['model1'].train(train_df, val_df, epochs=30)
                self.models['model1'].save()
                print("  ✅ Обучена")
            else:
                print("  ⚠️ Недостаточно данных")

        # Model 2
        if 'model2' in self.data:
            print("\n2️⃣ Model 2: Cross-Region")
            train_df = self.data['model2'].get('train')

            if train_df is not None and len(train_df) > 50:
                self.models['model2'].train(train_df)
                self.models['model2'].save('./models/region_model.pkl')
                print("  ✅ Обучена")
            else:
                print("  ⚠️ Недостаточно данных")

        print("\n✅ Все модели обучены!")

    def evaluate(self):
        """Оценка моделей"""
        print("\n📊 Оцениваю модели...")

        results = {}

        # Model 1
        if 'model1' in self.data and 'test' in self.data['model1']:
            test_df = self.data['model1']['test']

            if len(test_df) > 0:
                predictions = self.models['model1'].predict(test_df)

                from sklearn.metrics import roc_auc_score, accuracy_score

                y_true = test_df['target_will_purchase'].values
                y_pred_prob = predictions['purchase_probability']
                y_pred = predictions['will_purchase']

                auc = roc_auc_score(y_true, y_pred_prob)
                accuracy = accuracy_score(y_true, y_pred)

                results['model1'] = {
                    'auc': float(auc),
                    'accuracy': float(accuracy)
                }
                print(f"  Model 1: AUC={auc:.3f}, Accuracy={accuracy:.3f}")

        # Model 2
        if 'model2' in self.data and 'test' in self.data['model2']:
            test_df = self.data['model2']['test']

            if len(test_df) > 0:
                predictions = self.models['model2'].predict(test_df)

                from sklearn.metrics import mean_absolute_error

                y_true = []
                y_pred = []

                for region in test_df['region'].unique():
                    region_true = test_df[test_df['region'] == region]['target_purchase_count'].values
                    if region in predictions:
                        region_pred = predictions[region]['predicted_demand']
                        y_true.extend(region_true)
                        y_pred.extend([region_pred] * len(region_true))

                if y_true and y_pred:
                    mae = mean_absolute_error(y_true, y_pred)
                    results['model2'] = {'mae': float(mae)}
                    print(f"  Model 2: MAE={mae:.2f}")

        return results

    def show_predictions(self):
        """Показать примеры предсказаний"""
        print("\n🔮 Примеры предсказаний:")

        # Model 1
        if 'model1' in self.data and 'test' in self.data['model1']:
            try:
                test_df = self.data['model1']['test'].head(10)
                predictions = self.models['model1'].predict(test_df)
                print(f"\nModel 1 (первые 10 пользователя):")
                if 'purchase_probability' in predictions:
                    for i in range(min(10, len(predictions['purchase_probability']))):
                        prob = predictions['purchase_probability'][i]
                        print(f"  Пользователь {i + 1}: вероятность покупки = {prob:.1%}")
            except Exception as e:
                print(f"  ⚠️ Ошибка Model 1: {e}")

        # Model 2
        if 'model2' in self.data and 'test' in self.data['model2']:
            try:
                test_df = self.data['model2']['test']
                predictions = self.models['model2'].predict(test_df)
                print(f"\nModel 2 (по регионам):")
                for region, pred in list(predictions.items())[:3]:  # Только первые 3 региона
                    if isinstance(pred, dict):
                        demand = pred.get('predicted_demand', 0)
                        print(f"  {region}: спрос = {demand:.1f} единиц")
            except Exception as e:
                print(f"  ⚠️ Ошибка Model 2: {e}")

def main():
    print("=" * 60)
    print("🏗️  ПАЙПЛАЙН МОДЕЛЕЙ")
    print("=" * 60)

    # 1. Инициализация
    pipeline = ModelPipeline()

    # 2. Загрузка данных
    pipeline.load_snapshots()

    # 3. Обучение моделей
    pipeline.train_all()

    # 4. Оценка
    results = pipeline.evaluate()

    # 5. Примеры предсказаний
    pipeline.show_predictions()

    print("\n" + "=" * 60)
    print("✅ ВСЁ ГОТОВО!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()