# innovative_models/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class UserData(BaseModel):
    user_id: str
    features: Dict[str, Any]


class RegionData(BaseModel):
    region: str
    features: Dict[str, Any]


class TrendData(BaseModel):
    trend_id: str
    snapshot_date: str
    features: Dict[str, Any]


class ItemData(BaseModel):
    item_id: str
    current_price: float
    features: Dict[str, Any]


class MarketContext(BaseModel):
    competition_price: Optional[float] = None
    competition_weight: Optional[float] = 0.3
    market_segment: Optional[str] = "standard"
    supply_demand_ratio: Optional[float] = 1.0


class PredictionRequest(BaseModel):
    user_data: Optional[UserData] = None
    region_data: Optional[RegionData] = None
    trend_data: Optional[List[TrendData]] = None
    item_data: Optional[ItemData] = None
    market_context: Optional[MarketContext] = None