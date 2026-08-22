from pathlib import Path

import pytest
from pydantic import ValidationError

from ccexchange.config import RuntimeSettings, load_config


def test_defaults_are_paper_and_dry_run(monkeypatch):
    monkeypatch.chdir("/")
    s = RuntimeSettings(_env_file=None)
    assert s.paper_trading and s.dry_run and not s.live_trading
    s.assert_execution_safe()


def test_live_requires_all_gates():
    with pytest.raises(ValueError):
        RuntimeSettings(
            live_trading=True, paper_trading=False, dry_run=False, live_trading_acknowledgement="no"
        ).assert_execution_safe()
    RuntimeSettings(
        live_trading=True,
        paper_trading=False,
        dry_run=False,
        live_trading_acknowledgement="I_UNDERSTAND_LIVE_CRYPTO_ORDERS",
    ).assert_execution_safe()


def test_no_ambiguous_nonpaper_execution():
    with pytest.raises(ValueError):
        RuntimeSettings(paper_trading=False, dry_run=False).assert_execution_safe()


def test_alpaca_paper_environment_alias(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    settings = RuntimeSettings(_env_file=None)
    assert settings.paper_trading is False


def test_strategy_config_rejects_stock_ticker(tmp_path):
    source = Path("config/default.yaml").read_text(encoding="utf-8")
    path = tmp_path / "stock-universe.yaml"
    path.write_text(source.replace("[BTC/USD, ETH/USD]", "[BTC/USD, SPY]"), encoding="utf-8")

    with pytest.raises(ValidationError, match="BASE/USD"):
        load_config(path)
