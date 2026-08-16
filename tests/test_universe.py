from ccexchange.config import load_config
from ccexchange.universe import STARTING_UNIVERSE, RankedCandidate, rank_candidates


def test_starting_universe_is_only_btc_and_eth():
    cfg = load_config("config/default.yaml")
    assert tuple(cfg.symbols) == STARTING_UNIVERSE == ("BTC/USD", "ETH/USD")


def test_liquidity_gate_precedes_momentum_ranking():
    ranked = rank_candidates(
        [
            RankedCandidate("BTC/USD", 72, 10_000_000, True),
            RankedCandidate("ETH/USD", 84, 8_000_000, True),
            RankedCandidate("TINY/USD", 99, 100, False),
        ]
    )
    assert [candidate.symbol for candidate in ranked] == ["ETH/USD", "BTC/USD"]
