import pytest

from ccexchange.config import RuntimeSettings


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
