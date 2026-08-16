from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd


class MarketData(ABC):
    @abstractmethod
    def bars(
        self, symbols: list[str], timeframe: str, start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]: ...


def completed_only(frame: pd.DataFrame, candle_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Keep bars whose index is a close timestamp at or before the decision time."""
    now = candle_end or pd.Timestamp(datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    idx = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index
    return frame.loc[idx <= now].copy()


def resample_completed_hourly(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Build right-labeled 4h/12h bars; callers still apply completed_only at decision time."""
    if hours not in (4, 12):
        raise ValueError("only 4-hour and 12-hour aggregation is supported")
    rule = f"{hours}h"
    return (
        frame.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


class AlpacaCryptoData(MarketData):
    def __init__(self, api_key: str, secret_key: str):
        from alpaca.data.historical import CryptoHistoricalDataClient

        self.client = CryptoHistoricalDataClient(api_key, secret_key)

    def bars(self, symbols, timeframe, start, end):
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf = {"1Day": TimeFrame.Day, "1Hour": TimeFrame.Hour}.get(timeframe)
        aggregate = (
            int(timeframe.removesuffix("Hour")) if timeframe in {"4Hour", "12Hour"} else None
        )
        if aggregate:
            tf = TimeFrame.Hour
        if tf is None:
            raise ValueError("timeframe must be 1Day, 1Hour, 4Hour, or 12Hour")
        raw = self.client.get_crypto_bars(
            CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end)
        ).df
        frames = {s: raw.xs(s).rename(columns=str.lower) for s in symbols}
        # Alpaca labels native bars by opening time. Internally every bar is close-labeled,
        # making it impossible for the runtime to consume a still-forming candle.
        native_delta = (
            pd.Timedelta(hours=1) if aggregate or timeframe == "1Hour" else pd.Timedelta(days=1)
        )
        for frame in frames.values():
            frame.index = frame.index + native_delta
        return (
            {s: resample_completed_hourly(f, aggregate) for s, f in frames.items()}
            if aggregate
            else frames
        )
