from ccexchange.config import Risk
from ccexchange.risk import ratchet_stop, size_position


def test_position_is_risk_and_exposure_capped():
    cfg = Risk(risk_per_trade=0.01, atr_stop_multiplier=2, max_position_percent=0.2)
    plan = size_position(100_000, 100, 5, 50_000, cfg)
    assert plan.initial_stop == 90
    assert plan.risk_dollars <= 1000
    assert plan.notional <= 20_000


def test_stop_never_moves_backwards_when_atr_expands():
    cfg = Risk(atr_stop_multiplier=2)
    assert ratchet_stop(95, 110, 20, cfg) == 95
    assert ratchet_stop(95, 110, 5, cfg) == 100
