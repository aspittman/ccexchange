from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Indicators


def add_indicators(
    bars: pd.DataFrame, cfg: Indicators, annualization: float = 365.0
) -> pd.DataFrame:
    """Calculate indicators from OHLCV. Rows must be completed candles only."""
    x = bars.sort_index().copy()
    c, h, l = x.close, x.high, x.low
    for period, name in [
        (cfg.ema_fast, "ema_fast"),
        (cfg.ema_slow, "ema_slow"),
        (cfg.ema_long, "ema_long"),
    ]:
        x[name] = c.ewm(span=period, adjust=False).mean()
    fast = c.ewm(span=cfg.macd_fast, adjust=False).mean()
    slow = c.ewm(span=cfg.macd_slow, adjust=False).mean()
    x["macd"] = fast - slow
    x["macd_signal"] = x.macd.ewm(span=cfg.macd_signal, adjust=False).mean()
    x["macd_hist"] = x.macd - x.macd_signal
    x["macd_acceleration"] = x.macd_hist.diff()
    prev = c.shift()
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / cfg.adx_period, adjust=False).mean() / x.atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / cfg.adx_period, adjust=False).mean() / x.atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx"] = dx.ewm(alpha=1 / cfg.adx_period, adjust=False).mean()
    mid = c.rolling(cfg.bollinger_period).mean()
    std = c.rolling(cfg.bollinger_period).std(ddof=0)
    x["bb_mid"], x["bb_upper"], x["bb_lower"] = (
        mid,
        mid + cfg.bollinger_std * std,
        mid - cfg.bollinger_std * std,
    )
    x["bb_width"] = (x.bb_upper - x.bb_lower) / mid
    x["bb_width_percentile"] = x.bb_width.rolling(cfg.bandwidth_lookback).rank(pct=True)
    x["volume_ratio"] = x.volume / x.volume.rolling(cfg.volume_period).mean()
    x["dollar_volume"] = c * x.volume
    x["realized_volatility"] = c.pct_change().rolling(30).std() * np.sqrt(annualization)
    x["drawdown"] = c / c.cummax() - 1
    return x
