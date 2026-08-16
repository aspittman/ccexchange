from __future__ import annotations

from dataclasses import dataclass

# Deliberate allowlist: adding a symbol here or in config is an operator decision.
# The bot never discovers or trades arbitrary tokens automatically.
STARTING_UNIVERSE = ("BTC/USD", "ETH/USD")


@dataclass(frozen=True)
class RankedCandidate:
    symbol: str
    momentum_score: float
    dollar_volume: float
    eligible: bool
    reason: str = ""


def rank_candidates(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Liquidity is a hard gate; momentum only ranks assets that pass it."""
    return sorted(
        (candidate for candidate in candidates if candidate.eligible),
        key=lambda candidate: (candidate.momentum_score, candidate.dollar_volume),
        reverse=True,
    )
