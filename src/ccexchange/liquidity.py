"""Paper-entry quote checks. Displayed size is a cap, not a fill guarantee."""
from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class EntryLiquidity:
    ask: float
    max_notional: float
    spread: float


def entry_liquidity(quote, now, cfg):
    bid, ask = float(quote['price']), float(quote['ask'])
    ask_size = float(quote['ask_size'])
    bid_size = float(quote['bid_size'])
    if not all(math.isfinite(v) and v > 0 for v in (bid, ask, ask_size, bid_size)) or ask < bid:
        raise ValueError('Invalid entry quote or missing displayed size')
    stamp = pd.Timestamp(quote['timestamp'])
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('Unknown quote timestamp')
    age = (pd.Timestamp(now) - stamp).total_seconds()
    if not -5 <= age <= cfg.max_quote_age_seconds:
        raise ValueError('Stale or future entry quote')
    spread = (ask - bid) / ((ask + bid) / 2)
    if spread > cfg.max_entry_spread:
        raise ValueError('Entry spread exceeds limit')
    limit = ask * ask_size * cfg.max_ask_size_fraction
    if limit < cfg.minimum_entry_notional:
        raise ValueError('Insufficient displayed ask size')
    return EntryLiquidity(ask, limit, spread)
