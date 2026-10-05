from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Indicators(BaseModel):
    ema_fast: int = 20
    ema_slow: int = 50
    ema_long: int = 200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    adx_threshold: float = 22
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    bandwidth_lookback: int = 60
    atr_period: int = 14
    volume_period: int = 20
    relative_strength_period: int = 20


class Score(BaseModel):
    entry_threshold: float = 65
    weights: dict[str, float]

    @model_validator(mode="after")
    def valid_weights(self):
        if any(v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("score weights must be non-negative and have a positive sum")
        return self


class Regime(BaseModel):
    strong_uptrend_score: float = 75
    uptrend_score: float = 55
    downtrend_score: float = 35
    crash_drawdown: float = 0.20
    crash_realized_volatility: float = 1.20
    sideways_entry_multiplier: float = 1.2
    downtrend_entry_multiplier: float = 1.5


class Risk(BaseModel):
    risk_per_trade: float = 0.005
    atr_stop_multiplier: float = 2.5
    emergency_stop_percent: float = 0.15
    max_position_percent: float = 0.30
    max_crypto_exposure: float = 0.60
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    suspension_hours: int = 24
    max_positions: int = 2
    symbol_cooldown_bars: int = 1
    loss_cooldown_bars: int = 3
    correlation_lookback: int = 60
    max_pairwise_correlation: float = 0.80
    max_correlated_exposure: float = 0.35

    @model_validator(mode="after")
    def valid_limits(self):
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if min(self.symbol_cooldown_bars, self.loss_cooldown_bars) < 0:
            raise ValueError("cooldown bars cannot be negative")
        if self.correlation_lookback < 2:
            raise ValueError("correlation_lookback must be at least 2")
        if not 0 <= self.max_pairwise_correlation <= 1:
            raise ValueError("max_pairwise_correlation must be between 0 and 1")
        if not 0 < self.max_correlated_exposure <= 1:
            raise ValueError("max_correlated_exposure must be in (0, 1]")
        return self


class Backtest(BaseModel):
    initial_cash: float = 100_000
    fee_bps: float = 15
    slippage_bps: float = 10


class Liquidity(BaseModel):
    minimum_dollar_volume: float = 1_000_000


class StrategyConfig(BaseModel):
    symbols: list[str] = ["BTC/USD", "ETH/USD"]
    benchmark: str = "BTC/USD"
    timeframe: str = "1Day"
    max_new_positions_per_cycle: int = 1
    lookback_bars: int = 400
    minimum_history_bars: int = 220
    liquidity: Liquidity = Liquidity()
    indicators: Indicators = Indicators()
    score: Score
    regime: Regime = Regime()
    risk: Risk = Risk()
    backtest: Backtest = Backtest()

    @model_validator(mode="after")
    def valid_universe(self):
        if self.benchmark not in self.symbols:
            raise ValueError("benchmark must be included in symbols")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("universe symbols must be unique")
        invalid = [
            symbol
            for symbol in self.symbols
            if len(symbol.split("/")) != 2
            or not symbol.split("/")[0]
            or symbol.split("/")[1] != "USD"
        ]
        if invalid:
            raise ValueError(f"crypto universe symbols must use BASE/USD format: {invalid}")
        if self.max_new_positions_per_cycle < 1:
            raise ValueError("max_new_positions_per_cycle must be positive")
        return self


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    paper_trading: bool = Field(
        default=True,
        validation_alias=AliasChoices("PAPER_TRADING", "ALPACA_PAPER"),
    )
    live_trading: bool = False
    dry_run: bool = True
    live_trading_acknowledgement: str = ""
    config_path: str = "config/default.yaml"

    def assert_execution_safe(self) -> None:
        if self.live_trading:
            if self.paper_trading or self.dry_run:
                raise ValueError("live mode cannot be combined with paper or dry-run mode")
            if self.live_trading_acknowledgement != "I_UNDERSTAND_LIVE_CRYPTO_ORDERS":
                raise ValueError("live trading requires the exact acknowledgement phrase")
        elif not self.paper_trading and not self.dry_run:
            raise ValueError("unsafe mode: enable paper trading or dry-run")


def load_config(path: str | Path) -> StrategyConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return StrategyConfig.model_validate(raw)
