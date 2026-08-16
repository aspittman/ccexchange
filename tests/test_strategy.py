import pandas as pd

from ccexchange.config import load_config
from ccexchange.regime import MarketRegime, classify_regime
from ccexchange.scoring import momentum_score


def test_explainable_weighted_score():
    cfg = load_config("config/default.yaml")
    row = pd.Series(
        {
            "close": 120,
            "ema_fast": 115,
            "ema_slow": 110,
            "ema_long": 100,
            "macd": 3,
            "macd_signal": 2,
            "macd_acceleration": 0.2,
            "adx": 30,
            "bb_upper": 118,
            "bb_width_percentile": 0.8,
            "bb_width": 0.10,
            "bb_width_prev": 0.08,
            "volume_ratio": 1.5,
        }
    )
    result = momentum_score(row, cfg, relative=0.1)
    assert result.total == sum(cfg.score.weights.values())
    assert result.components["macd_momentum"] == cfg.score.weights["macd_momentum"]


def test_crash_regime_blocks_first():
    cfg = load_config("config/default.yaml")
    row = pd.Series(
        {
            "drawdown": -0.25,
            "realized_volatility": 0.5,
            "close": 90,
            "ema_long": 100,
            "ema_fast": 90,
            "ema_slow": 95,
            "macd": -1,
            "macd_signal": 0,
            "adx": 40,
            "macd_acceleration": -1,
        }
    )
    assert classify_regime(row, cfg)[0] == MarketRegime.HIGH_RISK
