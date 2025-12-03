# innovative_models/pipelines/production_pipeline.py
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from models.models.context_aware import ContextAwareModel
from models.models.cross_region import CrossRegionModel
from models.models.micro_trend import MicroTrendModel
from models.models.adaptive_pricing import AdaptivePricingModel


class ProductionPipeline:
    """Продакшен пайплайн для всех моделей"""

    def __init__(self, models_dir: str = "../production_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.models = {
            'context_aware': ContextAwareModel(),
            'cross_region': CrossRegionModel(),
            'micro_trend': MicroTrendModel(),
            'adaptive_pricing': AdaptivePricingModel()
        }

        self.is_trained = {name: False for name in self.models}
        self.metadata = {}

    def train_all_models(self, snapshots_dir: str = "../../analytics/data/innovative_snapshots"):
        """Обучение всех моделей в production режиме"""

        print("=" * 80)
        print("🚀 PRODUCTION TRAINING OF ALL 4 REAL MODELS")
        print("=" * 80)

        snapshots_path = Path(snapshots_dir)

        # Model 1: Context-Aware
        print("\n" + "=" * 40)
        print("1️⃣ REAL Context-Aware Purchase Prediction")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model1/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model1/val.parquet")

            # Фильтруем для тестирования (можно убрать)
            train_df = train_df.head(10000)
            val_df = val_df.head(2000)

            self.models['context_aware'].train(train_df, val_df, epochs=50)

            # Сохраняем модель
            torch.save({
                'model_state': self.models['context_aware'].model.state_dict(),
                'feature_config': self.models['context_aware'].feature_config,
                'scalers': self.models['context_aware'].scalers,
                'encoders': self.models['context_aware'].encoders,
                'category_encoder': getattr(self.models['context_aware'], 'category_encoder', None),
                'calibrator': self.models['context_aware'].calibrator
            }, self.models_dir / "context_model.pt")

            self.is_trained['context_aware'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

        # Model 2: Cross-Region
        print("\n" + "=" * 40)
        print("2️⃣ REAL Cross-Region Demand Transfer")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model2/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model2/val.parquet")

            train_df = train_df.head(5000)
            val_df = val_df.head(1000)

            self.models['cross_region'].train(train_df, val_df, epochs=100)

            torch.save({
                'model_state': self.models['cross_region'].model.state_dict(),
                'region_encoder': self.models['cross_region'].region_encoder,
                'feature_stats': getattr(self.models['cross_region'], 'feature_stats', None)
            }, self.models_dir / "region_model.pt")

            self.is_trained['cross_region'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Model 3: Micro-Trend
        print("\n" + "=" * 40)
        print("3️⃣ REAL Micro-Trend Anticipation")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model3/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model3/val.parquet")

            train_df = train_df.head(10000)
            val_df = val_df.head(2000)

            self.models['micro_trend'].train(train_df, val_df, epochs=50)

            torch.save({
                'model_state': self.models['micro_trend'].model.state_dict(),
                'scaler': self.models['micro_trend'].scaler
            }, self.models_dir / "trend_model.pt")

            self.is_trained['micro_trend'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Model 4: Adaptive Pricing
        print("\n" + "=" * 40)
        print("4️⃣ REAL Adaptive Pricing with RL")
        print("=" * 40)
        try:
            train_df = pd.read_parquet(snapshots_path / "model4/train.parquet")
            val_df = pd.read_parquet(snapshots_path / "model4/val.parquet")

            train_df = train_df.head(8000)
            val_df = val_df.head(1500)

            self.models['adaptive_pricing'].train(train_df, val_df, epochs=80)

            # Сохраняем модель спроса
            torch.save({
                'demand_model_state': self.models['adaptive_pricing'].demand_model.state_dict(),
                'scaler': self.models['adaptive_pricing'].scaler,
                'feature_names': getattr(self.models['adaptive_pricing'], 'feature_names', None),
                'price_bins': self.models['adaptive_pricing'].price_bins
            }, self.models_dir / "pricing_model.pt")

            # Сохраняем RL агента
            if self.models['adaptive_pricing'].rl_agent:
                torch.save(
                    self.models['adaptive_pricing'].rl_agent.policy_net.state_dict(),
                    self.models_dir / "rl_policy.pt"
                )

            self.is_trained['adaptive_pricing'] = True
            print("   ✅ Model trained and saved")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        # Сохраняем метаданные
        self.metadata = {
            'trained_models': [name for name, trained in self.is_trained.items() if trained],
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'context_aware': '2.0',
                'cross_region': '2.0',
                'micro_trend': '2.0',
                'adaptive_pricing': '2.0'
            },
            'training_stats': {
                'context_aware': {
                    'samples': len(train_df) if 'context_aware' in locals() else 0
                }
            }
        }

        with open(self.models_dir / "production_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print("\n" + "=" * 80)
        trained_count = sum(self.is_trained.values())
        print(f"✅ {trained_count}/4 REAL MODELS TRAINED FOR PRODUCTION!")
        print("=" * 80)

    def load_all_models(self):
        """Загрузка всех моделей"""
        print("🔄 Loading production models...")

        # Model 1
        try:
            checkpoint = torch.load(self.models_dir / "context_model.pt",
                                    map_location=self.models['context_aware'].device)

            # Recreate model
            self.models['context_aware'].feature_config = checkpoint['feature_config']
            self.models['context_aware'].model = MultiModalTransformer(
                num_numerical_features=len(checkpoint['feature_config']['numerical_features']),
                categorical_dims=checkpoint['feature_config']['categorical_features']
            ).to(self.models['context_aware'].device)

            self.models['context_aware'].model.load_state_dict(checkpoint['model_state'])
            self.models['context_aware'].scalers = checkpoint['scalers']
            self.models['context_aware'].encoders = checkpoint['encoders']
            self.models['context_aware'].category_encoder = checkpoint['category_encoder']
            self.models['context_aware'].calibrator = checkpoint['calibrator']

            self.is_trained['context_aware'] = True
            print("  ✅ Context-Aware: Loaded")
        except Exception as e:
            print(f"  ❌ Context-Aware: {e}")

        # Model 2
        try:
            checkpoint = torch.load(self.models_dir / "region_model.pt",
                                    map_location=self.models['cross_region'].device)

            # Recreate model
            node_features = len(checkpoint['feature_stats']['mean']) if checkpoint['feature_stats'] else 50
            self.models['cross_region'].model = GraphAttentionNetwork(node_features=node_features)
            self.models['cross_region'].model.load_state_dict(checkpoint['model_state'])
            self.models['cross_region'].region_encoder = checkpoint['region_encoder']

            self.is_trained['cross_region'] = True
            print("  ✅ Cross-Region: Loaded")
        except Exception as e:
            print(f"  ❌ Cross-Region: {e}")

        # Model 3
        try:
            checkpoint = torch.load(self.models_dir / "trend_model.pt",
                                    map_location=self.models['micro_trend'].device)

            self.models['micro_trend'].model = MicroTrendModel()
            self.models['micro_trend'].model.load_state_dict(checkpoint['model_state'])
            self.models['micro_trend'].scaler = checkpoint['scaler']

            self.is_trained['micro_trend'] = True
            print("  ✅ Micro-Trend: Loaded")
        except Exception as e:
            print(f"  ❌ Micro-Trend: {e}")

        # Model 4
        try:
            checkpoint = torch.load(self.models_dir / "pricing_model.pt",
                                    map_location=self.models['adaptive_pricing'].device)

            # Recreate model
            feature_dim = len(checkpoint['feature_names']) if checkpoint['feature_names'] else 10
            self.models['adaptive_pricing'].demand_model = DemandTransformer(feature_dim=feature_dim)
            self.models['adaptive_pricing'].demand_model.load_state_dict(checkpoint['demand_model_state'])
            self.models['adaptive_pricing'].scaler = checkpoint['scaler']

            # Load RL agent
            rl_path = self.models_dir / "rl_policy.pt"
            if rl_path.exists():
                state_dim = feature_dim + 3
                self.models['adaptive_pricing'].rl_agent = ReinforcementLearningPricer(
                    state_dim,
                    len(checkpoint['price_bins'])
                )
                self.models['adaptive_pricing'].rl_agent.policy_net.load_state_dict(
                    torch.load(rl_path)
                )

            self.is_trained['adaptive_pricing'] = True
            print("  ✅ Adaptive Pricing: Loaded")
        except Exception as e:
            print(f"  ❌ Adaptive Pricing: {e}")

    def get_api_predictions(self, request_data: Dict) -> Dict:
        """API endpoint для всех предсказаний"""

        results = {
            'context_aware': None,
            'cross_region': None,
            'micro_trend': None,
            'adaptive_pricing': None,
            'metadata': self.metadata
        }

        # Model 1: User prediction
        if 'user_data' in request_data and self.is_trained['context_aware']:
            try:
                user_df = pd.DataFrame([request_data['user_data']])
                results['context_aware'] = self.models['context_aware'].predict(user_df)
            except Exception as e:
                results['context_aware'] = {'error': str(e)}

        # Model 2: Region prediction
        if 'region_data' in request_data and self.is_trained['cross_region']:
            try:
                region_df = pd.DataFrame([request_data['region_data']])
                results['cross_region'] = self.models['cross_region'].predict(region_df)
            except Exception as e:
                results['cross_region'] = {'error': str(e)}

        # Model 3: Trend prediction
        if 'trend_data' in request_data and self.is_trained['micro_trend']:
            try:
                trend_df = pd.DataFrame(request_data['trend_data'])
                results['micro_trend'] = self.models['micro_trend'].predict(trend_df)
            except Exception as e:
                results['micro_trend'] = {'error': str(e)}

        # Model 4: Price recommendation
        if 'item_data' in request_data and self.is_trained['adaptive_pricing']:
            try:
                item_df = pd.DataFrame([request_data['item_data']])
                context = request_data.get('market_context', {})
                results['adaptive_pricing'] = self.models['adaptive_pricing'].recommend_price(
                    item_df, context
                )
            except Exception as e:
                results['adaptive_pricing'] = {'error': str(e)}

        return results
