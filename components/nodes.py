from __future__ import annotations
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

from .state import StockAgentState
from .indicators import sma, rsi
from . import config


# ---------- Helpers: tránh lỗi boolean với Series & ép scalar an toàn ----------
def _first_scalar(x):
    """Trả về scalar Python nếu x là pandas/NumPy scalar hoặc container 1 phần tử."""
    if isinstance(x, pd.Series):
        return x.iloc[0] if len(x) else None
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0] if len(x) else None
    if isinstance(x, np.generic):  # NumPy scalar
        return x.item()
    return x


def _is_missing(x):
    # Chuẩn hóa cách kiểm tra None/NaN
    return x is None or (isinstance(x, float) and pd.isna(x))


def _to_float_or_none(x):
    x = _first_scalar(x)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _ensure_errors(state: StockAgentState):
    if 'errors' not in state or state['errors'] is None:
        state['errors'] = []


def validate_ticker(state: StockAgentState) -> StockAgentState:
    ticker = state.get('ticker', '').strip().upper()
    _ensure_errors(state)
    if not ticker:
        state['is_valid'] = False
        state['errors'].append('Ticker is empty.')
        return state

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='1d')
        valid = (hist is not None and len(hist) > 0)
        if not valid:
            try:
                fi = getattr(t, 'fast_info', {})
                price = _first_scalar(fi.get('last_price', None))
                valid = (price is not None) and (float(price) > 0)
            except Exception:
                valid = False
        state['is_valid'] = bool(valid)
        if not valid:
            state['errors'].append(f'Không tìm thấy mã cổ phiếu: {ticker}')
    except Exception as e:
        state['is_valid'] = False
        state['errors'].append(f'Lỗi khi xác minh mã {ticker}: {e}')
    return state


def fetch_stock_data(state: StockAgentState) -> StockAgentState:
    ticker = state['ticker'].upper()
    _ensure_errors(state)
    try:
        t = yf.Ticker(ticker)

        # Lịch sử ~60 ngày (đặt auto_adjust để tránh FutureWarning)
        hist = yf.download(
            ticker,
            period=f"{config.HISTORY_DAYS}d",
            interval=config.DATA_INTERVAL,
            progress=False,
            auto_adjust=True,
        ).dropna()

        if hist.empty:
            state['errors'].append('Không lấy được dữ liệu lịch sử.')
            return state

        current_price = None
        market_cap = None
        currency = None
        long_name = None

        # Ưu tiên fast_info (nhanh)
        try:
            fi = t.fast_info
            current_price = _first_scalar(fi.get('last_price', None))
            market_cap = _first_scalar(fi.get('market_cap', None))
            currency = _first_scalar(fi.get('currency', None))
        except Exception:
            pass

        # Fallback: dùng last close nếu current_price chưa có
        if current_price is None:
            _last_close = hist['Close'].tail(1).squeeze()
            if not pd.isna(_last_close):
                current_price = float(_last_close)

        # Fallback thông tin công ty
        try:
            info = t.info
            if _is_missing(market_cap):
                market_cap = _first_scalar(info.get('marketCap'))
            if _is_missing(currency):
                currency = _first_scalar(info.get('currency'))
            _ln = _first_scalar(info.get('longName'))
            long_name = _ln if not _is_missing(_ln) else _first_scalar(info.get('shortName'))
        except Exception:
            pass

        volume = None
        if 'Volume' in hist.columns:
            v = hist['Volume'].tail(1).squeeze()
            if not pd.isna(v):
                volume = int(v)

        currency = currency or config.CURRENCY_FALLBACK

        state['stock_info'] = {
            'symbol': ticker,
            'long_name': long_name or ticker,
            'current_price': float(current_price) if current_price is not None else None,
            'market_cap': int(market_cap) if market_cap is not None and not pd.isna(market_cap) else None,
            'currency': currency,
            'last_volume': volume,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        state['historical_data'] = hist

    except Exception as e:
        state['errors'].append(f'Lỗi khi lấy dữ liệu cho {ticker}: {e}')
    return state


def calculate_technical_indicators(state: StockAgentState) -> StockAgentState:
    _ensure_errors(state)
    hist = state.get('historical_data')
    if hist is None or hist.empty:
        state['errors'].append('Thiếu dữ liệu lịch sử để tính chỉ báo.')
        return state

    try:
        close = hist['Close']
        # Nếu 'Close' trả về DataFrame (hiếm), ép về Series cột đầu tiên
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.astype(float)

        sma_short = sma(close, config.SMA_SHORT)
        sma_long = sma(close, config.SMA_LONG)
        rsi_series = rsi(close, config.RSI_PERIOD)

        df = hist.copy()
        df['SMA_SHORT'] = sma_short
        df['SMA_LONG'] = sma_long
        df['RSI'] = rsi_series
        state['historical_data'] = df

        latest = df.iloc[-1]
        tech = {
            'sma_short': _to_float_or_none(latest.get('SMA_SHORT')),
            'sma_long': _to_float_or_none(latest.get('SMA_LONG')),
            'rsi': _to_float_or_none(latest.get('RSI')),
            'last_close': _to_float_or_none(latest.get('Close')),
            'prev_close': _to_float_or_none(df['Close'].iloc[-2]) if len(df) >= 2 else None,
        }
        state['technical_indicators'] = tech
    except Exception as e:
        state['errors'].append(f'Lỗi khi tính chỉ báo: {e}')
    return state


def analyze_market_trend(state: StockAgentState) -> StockAgentState:
    _ensure_errors(state)
    tech = state.get('technical_indicators') or {}
    price = tech.get('last_close')
    sma_short_v = tech.get('sma_short')
    sma_long_v = tech.get('sma_long')
    rsi_v = tech.get('rsi')

    short_term = 'NEUTRAL'
    price_vs = 'Không xác định'
    momentum = 'NEUTRAL'
    notes = []

    if None not in (price, sma_short_v, sma_long_v):
        if sma_short_v > sma_long_v and price > sma_short_v:
            short_term = 'BULLISH ↗'
        elif sma_short_v < sma_long_v and price < sma_short_v:
            short_term = 'BEARISH ↘'
        else:
            short_term = 'NEUTRAL →'

        if price > sma_short_v and price > sma_long_v:
            price_vs = 'Above both moving averages'
        elif price < sma_short_v and price < sma_long_v:
            price_vs = 'Below both moving averages'
        else:
            price_vs = 'center both moving averages'

    prev_close = tech.get('prev_close')
    if prev_close is not None and price is not None:
        momentum = 'POSITIVE' if price >= prev_close else 'NEGATIVE'
    if rsi_v is not None:
        if rsi_v > 70:
            notes.append('RSI indicates overbought conditions')
        elif rsi_v < 30:
            notes.append('RSI indicates oversale conditions')

    state['trend'] = {
        'short_term': short_term,
        'price_vs_sma': price_vs,
        'momentum': momentum,
        'notes': '; '.join(notes) if notes else ''
    }
    return state


def _format_currency(value: float | int | None, currency: str) -> str:
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f"{currency} {value:,.2f}"
    return f"{currency} {value:,}"


def _format_market_cap(market_cap: int | None, currency: str) -> str:
    if market_cap is None:
        return 'N/A'
    caps = [
        (1_000_000_000_000, 'T'),
        (1_000_000_000, 'B'),
        (1_000_000, 'M'),
    ]
    for thresh, suffix in caps:
        if market_cap >= thresh:
            return f"{currency} {market_cap / thresh:.2f}{suffix}"
    return f"{currency} {market_cap:,}"


def generate_recommendation(state: StockAgentState) -> StockAgentState:
    _ensure_errors(state)

    if not state.get('is_valid', True):
        ticker = state.get('ticker', 'UNKNOWN')
        err = '; '.join(state.get('errors', []))
        state['report'] = f"Mã cổ phiếu không hợp lệ: {ticker}. Lỗi: {err}"
        return state

    info = state.get('stock_info') or {}
    tech = state.get('technical_indicators') or {}
    trend = state.get('trend') or {}

    symbol = info.get('symbol') or state.get('ticker', '')
    long_name = info.get('long_name') or symbol

    current_price = info.get('current_price') or tech.get('last_close')
    sma10 = tech.get('sma_short')
    sma20 = tech.get('sma_long')
    rsi_v = tech.get('rsi')

    decision = 'HOLD'
    reasons = []

    if None not in (current_price, sma10, sma20, rsi_v):
        if (current_price > sma10 > sma20) and (30 <= rsi_v <= 70):
            decision = 'BUY'
            reasons.append('price > SMA10 > SMA20 và RSI within 30-70 ')
        elif (current_price < sma20 < sma10) and (rsi_v > 70):
            decision = 'SELL'
            reasons.append('price < SMA20 < SMA10 và RSI > 70 ')
        else:
            decision = 'HOLD'
            reasons.append('not qualifify to buy/sale')
    else:
        reasons.append('Lack of sufficient indicator data to make a decision')

    currency = info.get('currency', 'USD')
    market_cap_fmt = _format_market_cap(info.get('market_cap'), currency)
    current_price_fmt = _format_currency(current_price, currency)

    analysis_time = info.get('analysis_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    lines = []
    lines.append('=' * 48)
    lines.append('Stock Analysis Report')
    lines.append('=' * 48)
    lines.append(f"stock: {long_name} ({symbol})")
    lines.append(f"current price: {current_price_fmt}")
    lines.append(f"maeket cap: {market_cap_fmt}")
    lines.append(f"analysis_time: {analysis_time}")
    lines.append('')
    lines.append('Technical Indicator')
    lines.append('-' * 20)
    lines.append(f"• SMA 10 days: {_format_currency(sma10, currency)}")
    lines.append(f"• SMA 20 day: {_format_currency(sma20, currency)}")
    lines.append(f"• RSI (14 days): {rsi_v:.2f}" if rsi_v is not None else "• RSI (14 day): N/A")
    lines.append('')
    lines.append('trend analysiss')
    lines.append('-' * 20)
    lines.append(f"• short trend: {trend.get('short_term', 'N/A')}")
    lines.append(f"• Price compared to SMA : {trend.get('price_vs_sma', 'N/A')}")
    lines.append(f"• Momentum: {trend.get('momentum', 'N/A')}")
    if trend.get('notes'):
        lines.append(f"• notes: {trend.get('notes')}")
    lines.append('')
    lines.append('Final Recommendation')
    lines.append('-' * 20)
    lines.append(f"• decision: {decision}")
    if reasons:
        lines.append(f"• reason: {', '.join(reasons)}")
    if state.get('errors'):
        lines.append('')
        lines.append(f"wraning: {'; '.join(state['errors'])}")

    state['report'] = ''.join(lines)
    return state
