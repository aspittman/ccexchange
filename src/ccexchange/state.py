from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ManagedPosition:
    symbol: str
    entry_price: float
    quantity: float
    stop: float
    high_watermark: float
    entry_time: str
    entry_score: float = 0.0
    regime: str = ""
    components: dict[str, float] = field(default_factory=dict)
    experiment_id: str = ""
    timeframe: str = ""


@dataclass
class BotState:
    positions: dict[str, ManagedPosition] = field(default_factory=dict)
    last_processed_candle: str = ""
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    equity_day: str = ""
    suspended_until: str = ""
    pending_orders: dict[str, str] = field(default_factory=dict)
    cooldowns: dict[str, int] = field(default_factory=dict)


class StateStore:
    def __init__(self, path: str | Path = "state/bot_state.json"):
        self.path = Path(path)

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["positions"] = {s: ManagedPosition(**p) for s, p in raw.get("positions", {}).items()}
        return BotState(**raw)

    def save(self, state: BotState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
