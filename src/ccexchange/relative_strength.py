import pandas as pd


def relative_strength(asset: pd.Series, benchmark: pd.Series, period: int) -> pd.Series:
    aligned = pd.concat([asset, benchmark], axis=1).ffill()
    return aligned.iloc[:, 0].pct_change(period) - aligned.iloc[:, 1].pct_change(period)


def btc_market_score(row: pd.Series) -> float:
    tests = [
        row.close > row.ema_fast,
        row.ema_fast > row.ema_slow,
        row.ema_slow > row.ema_long,
        row.macd > row.macd_signal,
    ]
    return sum(bool(v) for v in tests) / len(tests)
