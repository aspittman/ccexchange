from ccexchange.execution import DryRunBroker, OrderIntent


def test_dry_run_broker_never_reaches_alpaca():
    broker = DryRunBroker(50_000)
    result = broker.submit(OrderIntent("BTC/USD", "buy", 0.01, "test"))
    assert result["status"] == "dry_run"
    assert broker.account().equity == 50_000
    assert broker.orders[0].symbol == "BTC/USD"
