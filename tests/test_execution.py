from types import SimpleNamespace

import pytest

from ccexchange.execution import (
    AlpacaBroker,
    DryRunBroker,
    OrderIntent,
    deterministic_client_order_id,
)


def test_client_order_id_is_stable_and_decision_specific():
    first = deterministic_client_order_id("2025-01-01", "BTC/USD", "buy")
    assert first == deterministic_client_order_id("2025-01-01", "BTC/USD", "buy")
    assert first != deterministic_client_order_id("2025-01-02", "BTC/USD", "buy")


def test_dry_run_broker_never_reaches_alpaca():
    broker = DryRunBroker(50_000)
    result = broker.submit(OrderIntent("BTC/USD", "buy", 0.01, "test"))
    assert result["status"] == "dry_run"
    assert broker.account().equity == 50_000
    assert broker.orders[0].symbol == "BTC/USD"


def test_dry_run_broker_rejects_equity_symbols():
    broker = DryRunBroker()
    with pytest.raises(ValueError, match="non-crypto symbol"):
        broker.submit(OrderIntent("SPY", "buy", 1, "test"))
    assert broker.orders == []


def test_alpaca_broker_filters_equities_from_account():
    broker = object.__new__(AlpacaBroker)
    broker.client = SimpleNamespace(
        get_account=lambda: SimpleNamespace(equity="100000", cash="90000"),
        get_all_positions=lambda: [
            SimpleNamespace(
                symbol="SPY",
                asset_class="us_equity",
                qty="2",
                avg_entry_price="500",
                market_value="1000",
                current_price="500",
            ),
            SimpleNamespace(
                symbol="BTCUSD",
                asset_class="crypto",
                qty="0.1",
                avg_entry_price="90000",
                market_value="10000",
                current_price="100000",
            ),
        ],
    )

    account = broker.account()

    assert set(account.positions) == {"BTC/USD"}


def test_alpaca_broker_refuses_non_crypto_asset_before_submission():
    broker = object.__new__(AlpacaBroker)
    submitted = []
    broker.client = SimpleNamespace(
        get_asset=lambda _symbol: SimpleNamespace(asset_class="us_equity", tradable=True),
        submit_order=lambda **kwargs: submitted.append(kwargs),
    )

    with pytest.raises(ValueError, match="non-crypto symbol"):
        broker.submit(OrderIntent("SPY", "buy", 1, "test"))
    assert submitted == []


def test_alpaca_broker_refuses_disguised_non_crypto_asset():
    broker = object.__new__(AlpacaBroker)
    submitted = []
    broker.client = SimpleNamespace(
        get_asset=lambda _symbol: SimpleNamespace(asset_class="us_equity", tradable=True),
        submit_order=lambda **kwargs: submitted.append(kwargs),
    )

    with pytest.raises(ValueError, match="non-crypto order"):
        broker.submit(OrderIntent("SPY/USD", "buy", 1, "test"))
    assert submitted == []
