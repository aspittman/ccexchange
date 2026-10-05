from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any
from uuid import uuid4

from .config import RuntimeSettings


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    reason: str
    client_order_id: str = ""


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


@dataclass(frozen=True)
class OrderUpdate:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    requested_quantity: float
    filled_quantity: float
    submitted_at: str
    updated_at: str


class Broker(ABC):
    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def submit(self, order: OrderIntent) -> Any: ...

    def filled_orders(self, symbols: list[str] | None = None) -> list[FilledOrder]:
        return []

    def open_client_order_ids(self) -> set[str]:
        return set()

    def order_updates(self, symbols: list[str] | None = None) -> list[OrderUpdate]:
        return []


class DryRunBroker(Broker):
    def __init__(self, equity: float = 100_000):
        self.equity = equity
        self.orders: list[OrderIntent] = []

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(self.equity, self.equity, {})

    def submit(self, order: OrderIntent):
        _require_crypto_pair(order.symbol)
        self.orders.append(order)
        return {"id": f"dry-run-{len(self.orders)}", "status": "dry_run"}


class AlpacaBroker(Broker):
    def __init__(self, settings: RuntimeSettings):
        settings.assert_execution_safe()
        if settings.dry_run:
            raise ValueError("AlpacaBroker cannot be constructed in dry-run mode")
        from ccexchange.bot_ownership import TradingClient

        self.client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.paper_trading
        )

    def account(self) -> AccountSnapshot:
        account = self.client.get_account()
        positions = {}
        for item in self.client.get_all_positions():
            # This bot shares an Alpaca account with unrelated stock/ETF bots. Keep
            # those holdings outside its reconciliation and position-management view.
            if _asset_class_value(getattr(item, "asset_class", None)) != "crypto":
                continue
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

        _require_crypto_pair(order.symbol)
        asset = self.client.get_asset(order.symbol)
        if _asset_class_value(getattr(asset, "asset_class", None)) != "crypto":
            raise ValueError(f"refusing non-crypto order for {order.symbol}")
        if not bool(getattr(asset, "tradable", False)):
            raise ValueError(f"refusing order for non-tradable crypto asset {order.symbol}")
        quantity = floor_quantity(
            order.quantity,
            getattr(asset, "min_trade_increment", None),
        )
        minimum = Decimal(str(getattr(asset, "min_order_size", None) or "0"))
        if quantity <= 0 or Decimal(str(quantity)) < minimum:
            raise ValueError(
                f"order quantity for {order.symbol} is below Alpaca's minimum order size"
            )
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=quantity,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=order.client_order_id or f"ccexchange-{uuid4().hex}",
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

    def open_client_order_ids(self) -> set[str]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {
            str(item.client_order_id)
            for item in orders
            if str(getattr(item, "client_order_id", "")).startswith("ccexchange-")
        }

    def order_updates(self, symbols: list[str] | None = None) -> list[OrderUpdate]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, symbols=symbols)
        )
        updates = []
        for item in orders:
            client_id = str(getattr(item, "client_order_id", ""))
            if not client_id.startswith("ccexchange-"):
                continue
            submitted_at = getattr(item, "submitted_at", None) or getattr(item, "created_at", None)
            updated_at = (
                getattr(item, "updated_at", None)
                or getattr(item, "filled_at", None)
                or submitted_at
            )
            updates.append(
                OrderUpdate(
                    str(item.id),
                    client_id,
                    _display_symbol(str(item.symbol)),
                    str(item.side.value),
                    str(item.status.value),
                    float(item.qty or 0),
                    float(item.filled_qty or 0),
                    submitted_at.isoformat() if submitted_at else "",
                    updated_at.isoformat() if updated_at else "",
                )
            )
        return updates


def deterministic_client_order_id(candle: str, symbol: str, side: str) -> str:
    """Stable idempotency key for one strategy decision."""
    import hashlib

    key = f"{candle}|{symbol.upper()}|{side.lower()}"
    return f"ccexchange-{hashlib.sha256(key.encode()).hexdigest()[:32]}"


def floor_quantity(quantity: float, increment: Any = None) -> float:
    """Round toward zero so an exit can never request more than the held balance."""
    step = Decimal(str(increment or "0.000000001"))
    if step <= 0:
        raise ValueError("trade increment must be positive")
    value = Decimal(str(quantity))
    return float((value / step).to_integral_value(rounding=ROUND_DOWN) * step)


def _display_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD"):
        return f"{symbol[:-3]}/USD"
    return symbol


def _asset_class_value(asset_class: Any) -> str:
    """Return an Alpaca enum/string asset class in a stable lowercase form."""
    return str(getattr(asset_class, "value", asset_class) or "").lower()


def _require_crypto_pair(symbol: str) -> None:
    """Fail closed before any broker can receive an equity-style symbol."""
    parts = symbol.upper().split("/")
    if len(parts) != 2 or not parts[0] or parts[1] != "USD":
        raise ValueError(f"refusing non-crypto symbol {symbol}; expected BASE/USD")


def order_id(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("id", "unknown"))
    return str(getattr(result, "id", "unknown"))


def make_broker(settings: RuntimeSettings, dry_run_equity: float = 100_000) -> Broker:
    settings.assert_execution_safe()
    return DryRunBroker(dry_run_equity) if settings.dry_run else AlpacaBroker(settings)
