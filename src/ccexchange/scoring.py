from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import StrategyConfig


@dataclass(frozen=True)
class ScoreResult:
    total: float
    components: dict[str, float]
    reasons: list[str]


def momentum_score(
    row: pd.Series, cfg: StrategyConfig, relative: float = 0.0, is_btc: bool = False
) -> ScoreResult:
    w, i = cfg.score.weights, cfg.indicators
    conditions = {
        "ema_trend": row.close > row.ema_fast > row.ema_slow and row.ema_slow > row.ema_long,
        "macd_momentum": row.macd > row.macd_signal and row.macd > 0,
        "macd_acceleration": row.macd_acceleration > 0,
        "adx_strength": row.adx >= i.adx_threshold,
        "bollinger_breakout": row.close > row.bb_upper,
        "bollinger_volatility_shift": row.bb_width_percentile >= 0.7
        and row.bb_width > row.get("bb_width_prev", 0),
        "volume_confirmation": row.volume_ratio >= 1.2,
        "relative_strength": relative > 0 if not is_btc else row.close > row.ema_long,
    }
    components = {name: round(w.get(name, 0) if ok else 0.0, 2) for name, ok in conditions.items()}
    return ScoreResult(
        round(sum(components.values()), 2),
        components,
        [name.replace("_", " ") for name, ok in conditions.items() if ok],
    )
