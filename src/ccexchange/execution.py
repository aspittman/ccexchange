from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .config import RuntimeSettings


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    reason: str


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: float
    average_entry: float
    market_value: float
    current_price: float


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    positions: dict[str, BrokerPosition]


@dataclass(frozen=True)
class FilledOrder:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    filled_at: str


class Broker(ABC):
    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def submit(self, order: OrderIntent) -> Any: ...

    def filled_orders(self, symbols: list[str] | None = None) -> list[FilledOrder]:
        return []


class DryRunBroker(Broker):
    def __init__(self, equity: float = 100_000):
        self.equity = equity
        self.orders: list[OrderIntent] = []

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(self.equity, self.equity, {})

    def submit(self, order: OrderIntent):
        self.orders.append(order)
        return {"id": f"dry-run-{len(self.orders)}", "status": "dry_run"}


class AlpacaBroker(Broker):
    def __init__(self, settings: RuntimeSettings):
        settings.assert_execution_safe()
        if settings.dry_run:
            raise ValueError("AlpacaBroker cannot be constructed in dry-run mode")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.paper_trading
        )

    def account(self) -> AccountSnapshot:
        account = self.client.get_account()
        positions = {}
        for item in self.client.get_all_positions():
            normalized = _display_symbol(str(item.symbol))
            positions[normalized] = BrokerPosition(
                normalized,
                float(item.qty),
                float(item.avg_entry_price),
                abs(float(item.market_value)),
                float(item.current_price),
            )
        return AccountSnapshot(float(account.equity), float(account.cash), positions)

    def submit(self, order: OrderIntent):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=round(order.quantity, 8),
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=f"ccexchange-{uuid4().hex}",
        )
        return self.client.submit_order(order_data=request)

    def filled_orders(self, symbols: list[str] | None = None) -> list[FilledOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, symbols=symbols)
        )
        fills = []
        for item in orders:
            if not str(item.client_order_id).startswith("ccexchange-"):
                continue
            if not item.filled_at or not item.filled_qty or not item.filled_avg_price:
                continue
            fills.append(
                FilledOrder(
                    str(item.id),
                    _display_symbol(str(item.symbol)),
                    str(item.side.value),
                    float(item.filled_qty),
                    float(item.filled_avg_price),
                    item.filled_at.isoformat(),
                )
            )
        return fills


def _display_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD"):
        return f"{symbol[:-3]}/USD"
    return symbol


def order_id(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("id", "unknown"))
    return str(getattr(result, "id", "unknown"))


def make_broker(settings: RuntimeSettings, dry_run_equity: float = 100_000) -> Broker:
    settings.assert_execution_safe()
    return DryRunBroker(dry_run_equity) if settings.dry_run else AlpacaBroker(settings)
