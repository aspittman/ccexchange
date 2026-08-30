from enum import Enum

import pandas as pd

from .config import StrategyConfig


class MarketRegime(str, Enum):
    STRONG_UPTREND = "STRONG UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS / CHOPPY"
    DOWNTREND = "DOWNTREND"
    HIGH_RISK = "HIGH-RISK / CRASH"


def classify_regime(btc: pd.Series, cfg: StrategyConfig) -> tuple[MarketRegime, float]:
    r = cfg.regime
    # A large drawdown from an old high should not permanently lock a recovering
    # market out. Treat drawdown as a crash only while short-term price and MACD
    # momentum are both still deteriorating. Extreme realized volatility remains
    # an unconditional safety block.
    drawdown_crash = (
        btc.drawdown <= -r.crash_drawdown
        and btc.close < btc.ema_fast
        and btc.macd < btc.macd_signal
    )
    if drawdown_crash or btc.realized_volatility >= r.crash_realized_volatility:
        return MarketRegime.HIGH_RISK, 0
    points = 0.0
    points += 25 if btc.close > btc.ema_long else 0
    points += 20 if btc.ema_fast > btc.ema_slow > btc.ema_long else 0
    points += 20 if btc.macd > btc.macd_signal else 0
    points += 15 if btc.adx >= cfg.indicators.adx_threshold else 0
    points += 10 if btc.macd_acceleration > 0 else 0
    points += 10 if btc.drawdown > -0.10 else 0
    if points >= r.strong_uptrend_score:
        regime = MarketRegime.STRONG_UPTREND
    elif points >= r.uptrend_score:
        regime = MarketRegime.UPTREND
    elif points < r.downtrend_score:
        regime = MarketRegime.DOWNTREND
    else:
        regime = MarketRegime.SIDEWAYS
    return regime, points


def required_entry_score(regime: MarketRegime, cfg: StrategyConfig) -> float:
    if regime == MarketRegime.HIGH_RISK:
        return float("inf")
    if regime == MarketRegime.DOWNTREND:
        return cfg.score.entry_threshold * cfg.regime.downtrend_entry_multiplier
    if regime == MarketRegime.SIDEWAYS:
        return cfg.score.entry_threshold * cfg.regime.sideways_entry_multiplier
    return cfg.score.entry_threshold
