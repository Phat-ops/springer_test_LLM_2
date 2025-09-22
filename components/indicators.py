from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average.
    SMA(n) = (P1 + P2 + ... + Pn) / n
    """
    return series.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) using simple rolling averages for gains/losses.
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))
    Where averages are simple means over the lookback period.
    """
    # Coerce DataFrame -> Series if single column (or take first column)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0] if close.shape[1] >= 1 else close.squeeze()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.bfill()
