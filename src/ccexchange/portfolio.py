from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    quantity: float
    entry: float
    stop: float
    high: float
    entry_time: object = None
    mfe: float = 0
    mae: float = 0
    entry_fee: float = 0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(
            p.quantity * prices.get(s, p.entry) for s, p in self.positions.items()
        )

    def exposure(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices.get(s, p.entry) for s, p in self.positions.items())
