from dataclasses import dataclass

from .config import Risk


@dataclass
class PositionPlan:
    quantity: float
    notional: float
    initial_stop: float
    risk_dollars: float


def size_position(
    equity: float, entry: float, atr: float, exposure_available: float, cfg: Risk
) -> PositionPlan:
    stop = max(entry - cfg.atr_stop_multiplier * atr, entry * (1 - cfg.emergency_stop_percent))
    distance = entry - stop
    if min(equity, entry, atr, distance, exposure_available) <= 0:
        return PositionPlan(0, 0, stop, 0)
    risk_budget = equity * cfg.risk_per_trade
    cap = min(equity * cfg.max_position_percent, exposure_available)
    qty = min(risk_budget / distance, cap / entry)
    return PositionPlan(qty, qty * entry, stop, qty * distance)


def ratchet_stop(previous: float, high_watermark: float, atr: float, cfg: Risk) -> float:
    return max(previous, high_watermark - cfg.atr_stop_multiplier * atr)


@dataclass
class CircuitBreaker:
    starting_equity: float
    peak_equity: float
    day_start_equity: float

    def reason(self, equity: float, cfg: Risk) -> str | None:
        self.peak_equity = max(self.peak_equity, equity)
        if equity / self.day_start_equity - 1 <= -cfg.max_daily_loss:
            return "maximum daily loss"
        if equity / self.peak_equity - 1 <= -cfg.max_drawdown:
            return "maximum portfolio drawdown"
        return None
