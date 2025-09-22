from __future__ import annotations
from typing import TypedDict, Optional, Dict, List
import pandas as pd

class StockAgentState(TypedDict, total=False):
    ticker: str
    stock_info: Optional[Dict]
    historical_data: Optional[pd.DataFrame]
    technical_indicators: Optional[Dict]
    trend: Optional[Dict]
    report: Optional[str]
    is_valid: Optional[bool]
    errors: Optional[List[str]]
