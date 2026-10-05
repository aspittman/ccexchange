from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .audit import AuditLog
from .config import RuntimeSettings, StrategyConfig, load_config
from .data import AlpacaCryptoData, completed_only
from .execution import (
    Broker,
    FilledOrder,
    OrderIntent,
    deterministic_client_order_id,
    make_broker,
    order_id,
)
from .indicators import add_indicators
from .paper import PaperRecorder
from .regime import MarketRegime, classify_regime, required_entry_score
from .relative_strength import relative_strength
from .risk import correlated_exposure_available, ratchet_stop, size_position
from .scoring import momentum_score
from .state import BotState, ManagedPosition, StateStore
from .universe import RankedCandidate, rank_candidates

INTERVAL_SECONDS = {"1Day": 3600, "12Hour": 900, "4Hour": 300, "1Hour": 60}
PERIODS_PER_YEAR = {"1Day": 365, "12Hour": 730, "4Hour": 2190, "1Hour": 8760}
LOGGER = logging.getLogger("ccexchange.runtime")


def _bot_performance(
    fills: list[FilledOrder], current_prices: dict[str, float], starting_equity: float
) -> tuple[int, float, float]:
    """Calculate P&L using only fills attributed to this bot."""
    quantities: dict[str, float] = {}
    costs: dict[str, float] = {}
    realized_pnl = 0.0
    for fill in sorted(fills, key=lambda item: item.filled_at):
        quantity = quantities.get(fill.symbol, 0.0)
        cost = costs.get(fill.symbol, 0.0)
        if fill.side.lower() == "buy":
            quantities[fill.symbol] = quantity + fill.quantity
            costs[fill.symbol] = cost + fill.quantity * fill.price
            continue
        sold = min(fill.quantity, quantity)
        average_cost = cost / quantity if quantity > 0 else 0.0
        realized_pnl += sold * (fill.price - average_cost)
        remaining = max(0.0, quantity - sold)
        quantities[fill.symbol] = remaining
        costs[fill.symbol] = average_cost * remaining

    unrealized_pnl = sum(
        quantity * current_prices.get(symbol, costs[symbol] / quantity) - costs[symbol]
        for symbol, quantity in quantities.items()
        if quantity > 1e-12
    )
    total_pnl = realized_pnl + unrealized_pnl
    return (
        sum(quantity > 1e-12 for quantity in quantities.values()),
        total_pnl,
        (total_pnl / starting_equity * 100) if starting_equity > 0 else 0.0,
    )


def _iso(value) -> str:
    return pd.Timestamp(value).isoformat()


def _risk_block(state: BotState, equity: float, cfg: StrategyConfig, now: datetime) -> str | None:
    today = now.date().isoformat()
    if state.equity_day != today or state.day_start_equity <= 0:
        state.equity_day, state.day_start_equity = today, equity
    state.peak_equity = max(state.peak_equity, equity)
    reason = None
    if equity / state.day_start_equity - 1 <= -cfg.risk.max_daily_loss:
        reason = "maximum daily portfolio loss"
    elif equity / state.peak_equity - 1 <= -cfg.risk.max_drawdown:
        reason = "maximum portfolio drawdown"
    if reason:
        state.suspended_until = (now + timedelta(hours=cfg.risk.suspension_hours)).isoformat()
        return reason
    if state.suspended_until and now < datetime.fromisoformat(state.suspended_until):
        return f"trading suspended until {state.suspended_until}"
    if state.suspended_until:
        state.suspended_until = ""
    return None


def _reconcile(
    state: BotState,
    account,
    rows: dict[str, pd.Series],
    cfg: StrategyConfig,
    audit: AuditLog,
    now: datetime,
    candle: str,
) -> None:
    for symbol, side in list(state.pending_orders.items()):
        settled = (side == "buy" and symbol in account.positions) or (
            side == "sell" and symbol not in account.positions
        )
        if settled:
            audit.write("ORDER_RECONCILED", symbol=symbol, side=side)
            del state.pending_orders[symbol]
        elif state.last_processed_candle and state.last_processed_candle != candle:
            # Crypto market orders normally settle promptly. Do not permanently lock a
            # symbol after a rejected/cancelled order; broker positions remain authoritative.
            audit.write("STALE_PENDING_ORDER_CLEARED", symbol=symbol, side=side)
            del state.pending_orders[symbol]
    for symbol in list(state.positions):
        if symbol not in account.positions:
            audit.write("POSITION_CLOSED_RECONCILIATION", symbol=symbol)
            del state.positions[symbol]
    for symbol, actual in account.positions.items():
        if symbol not in cfg.symbols:
            continue
        if symbol in state.positions:
            managed = state.positions[symbol]
            managed.quantity = actual.quantity
            managed.entry_price = actual.average_entry
            managed.high_watermark = max(managed.high_watermark, actual.current_price)
        elif symbol in rows:
            row = rows[symbol]
            stop = max(
                actual.average_entry - cfg.risk.atr_stop_multiplier * float(row.atr),
                actual.average_entry * (1 - cfg.risk.emergency_stop_percent),
            )
            state.positions[symbol] = ManagedPosition(
                symbol,
                actual.average_entry,
                actual.quantity,
                stop,
                max(actual.average_entry, actual.current_price),
                now.isoformat(),
            )
            audit.write(
                "POSITION_ADOPTED",
                symbol=symbol,
                quantity=actual.quantity,
                entry=actual.average_entry,
                stop=stop,
            )


def run_cycle(
    settings: RuntimeSettings,
    audit: AuditLog,
    broker: Broker | None = None,
    store: StateStore | None = None,
    source=None,
    now: datetime | None = None,
) -> None:
    cfg = load_config(settings.config_path)
    now = now or datetime.now(timezone.utc)
    hours = {"1Day": 24, "12Hour": 12, "4Hour": 4, "1Hour": 1}[cfg.timeframe]
    source = source or AlpacaCryptoData(settings.alpaca_api_key, settings.alpaca_secret_key)
    raw = source.bars(
        cfg.symbols, cfg.timeframe, now - timedelta(hours=hours * (cfg.lookback_bars + 5)), now
    )
    frames = {
        symbol: add_indicators(
            completed_only(frame, now), cfg.indicators, PERIODS_PER_YEAR[cfg.timeframe]
        )
        for symbol, frame in raw.items()
    }
    if any(len(frame) < cfg.minimum_history_bars for frame in frames.values()):
        audit.write(
            "DATA_NOT_READY",
            counts={s: len(f) for s, f in frames.items()},
            required=cfg.minimum_history_bars,
        )
        return
    latest = min(frame.index[-1] for frame in frames.values())
    frames = {symbol: frame.loc[:latest] for symbol, frame in frames.items()}
    rows = {symbol: frame.iloc[-1] for symbol, frame in frames.items()}
    candle = _iso(latest)
    store = store or StateStore()
    state = store.load()
    if state.last_processed_candle != candle:
        state.cooldowns = {
            symbol: bars - 1 for symbol, bars in state.cooldowns.items() if bars - 1 > 0
        }
    broker = broker or make_broker(settings, cfg.backtest.initial_cash)
    account = broker.account()
    open_client_order_ids = broker.open_client_order_ids() if not settings.dry_run else set()
    recorder = None
    experiment_id = ""
    bot_fills = broker.filled_orders(cfg.symbols) if not settings.dry_run else []
    order_updates = broker.order_updates(cfg.symbols) if not settings.dry_run else []
    if settings.paper_trading and not settings.dry_run:
        recorder = PaperRecorder()
        experiment_id = recorder.register_experiment(cfg.model_dump(mode="json"))
        recorder.record_bars(
            {
                symbol: frame[["open", "high", "low", "close", "volume"]]
                for symbol, frame in frames.items()
            },
            cfg.timeframe,
        )
        recorder.record_snapshot(now, account)
        recorder.record_fills(bot_fills)
        recorder.record_order_updates(order_updates)
        recorder.report()
    _reconcile(state, account, rows, cfg, audit, now, candle)
    bot_positions, bot_pnl, bot_return_percent = _bot_performance(
        bot_fills,
        {symbol: position.current_price for symbol, position in account.positions.items()},
        cfg.backtest.initial_cash,
    )
    audit.write(
        "PORTFOLIO_STATUS",
        bot_positions_held=bot_positions,
        bot_pnl=bot_pnl,
        bot_return_percent=bot_return_percent,
        basis="ccexchange-tagged fills versus configured initial cash",
    )
    btc = rows[cfg.benchmark]
    regime, regime_points = classify_regime(btc, cfg)
    audit.write("REGIME", regime=regime.value, score=regime_points, candle=candle)

    # Manage held positions on every poll. Exits are never blocked by an entry circuit breaker.
    for symbol, managed in list(state.positions.items()):
        if symbol not in rows or symbol not in account.positions:
            continue
        row = rows[symbol]
        current_price = account.positions[symbol].current_price
        managed.high_watermark = max(managed.high_watermark, float(row.high), current_price)
        managed.stop = ratchet_stop(managed.stop, managed.high_watermark, float(row.atr), cfg.risk)
        reason = None
        if current_price <= managed.stop:
            reason = "ratcheting ATR stop"
        elif current_price <= managed.entry_price * (1 - cfg.risk.emergency_stop_percent):
            reason = "emergency loss stop"
        elif regime == MarketRegime.HIGH_RISK:
            reason = "high-risk market regime"
        elif row.macd < row.macd_signal and row.close < row.ema_fast:
            reason = "MACD and EMA momentum deterioration"
        if reason and state.pending_orders.get(symbol) != "sell":
            client_id = deterministic_client_order_id(candle, symbol, "sell")
            if client_id in open_client_order_ids:
                state.pending_orders[symbol] = "sell"
                audit.write("DUPLICATE_ORDER_SUPPRESSED", symbol=symbol, side="sell", client_order_id=client_id)
                continue
            result = broker.submit(
                OrderIntent(
                    symbol,
                    "sell",
                    account.positions[symbol].quantity,
                    reason,
                    client_id,
                )
            )
            audit.write(
                "SELL_SUBMITTED",
                symbol=symbol,
                quantity=account.positions[symbol].quantity,
                reason=reason,
                order_id=order_id(result),
                stop=managed.stop,
                entry=managed.entry_price,
                current=current_price,
            )
            state.pending_orders[symbol] = "sell"
            state.cooldowns[symbol] = (
                cfg.risk.loss_cooldown_bars
                if current_price < managed.entry_price
                else cfg.risk.symbol_cooldown_bars
            ) + 1
            if recorder:
                recorder.record_order(
                    {
                        "order_id": order_id(result),
                        "timestamp": now.isoformat(),
                        "symbol": symbol,
                        "side": "sell",
                        "reason": reason,
                        "client_order_id": client_id,
                        "quantity": account.positions[symbol].quantity,
                        "signal_price": current_price,
                        "submitted_at": now.isoformat(),
                        "experiment_id": managed.experiment_id,
                        "timeframe": managed.timeframe,
                        "regime": regime.value,
                    }
                )
        else:
            audit.write(
                "POSITION_MANAGED",
                symbol=symbol,
                stop=managed.stop,
                high_watermark=managed.high_watermark,
                current=current_price,
            )

    # A completed candle can create at most one entry decision, even across restarts.
    if state.last_processed_candle == candle:
        store.save(state)
        return
    state.last_processed_candle = candle
    risk_reason = _risk_block(state, account.equity, cfg, now)
    if risk_reason:
        audit.write(
            "RISK_BLOCK",
            reason=risk_reason,
            equity=account.equity,
            peak_equity=state.peak_equity,
            day_start_equity=state.day_start_equity,
        )
        store.save(state)
        return
    current_exposure = sum(p.market_value for s, p in account.positions.items() if s in cfg.symbols)
    available = max(0.0, account.equity * cfg.risk.max_crypto_exposure - current_exposure)
    held_values = {
        symbol: position.market_value
        for symbol, position in account.positions.items()
        if symbol in cfg.symbols
    }
    recent_returns = {
        symbol: frame.close.pct_change().tail(cfg.risk.correlation_lookback)
        for symbol, frame in frames.items()
    }
    candidates = []
    candidate_details = {}
    scored_by_symbol = {}
    for symbol, frame in frames.items():
        row = rows[symbol]
        relative = (
            0.0
            if symbol == cfg.benchmark
            else float(
                relative_strength(
                    frame.close,
                    frames[cfg.benchmark].close,
                    cfg.indicators.relative_strength_period,
                ).iloc[-1]
            )
        )
        scored_row = row.copy()
        scored_row["bb_width_prev"] = frame.bb_width.iloc[-2]
        scored = momentum_score(scored_row, cfg, relative, symbol == cfg.benchmark)
        threshold = required_entry_score(regime, cfg)
        liquid = bool(row.dollar_volume >= cfg.liquidity.minimum_dollar_volume)
        eligible = bool(liquid and scored.total >= threshold)
        details = {
            "symbol": symbol,
            "candle": candle,
            "regime": regime.value,
            "score": scored.total,
            "threshold": threshold,
            "components": scored.components,
            "reasons": scored.reasons,
            "dollar_volume": float(row.dollar_volume),
            "eligible": eligible,
            "liquidity_eligible": liquid,
        }
        audit.write("SIGNAL", **details)
        if recorder:
            score_band = f"{int(scored.total // 10) * 10}-{int(scored.total // 10) * 10 + 9}"
            recorder.record_decision(
                {
                    "experiment_id": experiment_id,
                    "timestamp": now.isoformat(),
                    "candle": candle,
                    "symbol": symbol,
                    "timeframe": cfg.timeframe,
                    "regime": regime.value,
                    "score": scored.total,
                    "score_band": score_band,
                    "threshold": threshold,
                    "liquidity_eligible": liquid,
                    "eligible": eligible,
                    "dollar_volume": float(row.dollar_volume),
                    "adx": float(row.adx),
                    "atr": float(row.atr),
                    "volume_ratio": float(row.volume_ratio),
                    "relative_strength": relative,
                    **{f"component_{key}": value for key, value in scored.components.items()},
                }
            )
        candidates.append(
            RankedCandidate(
                symbol,
                scored.total,
                float(row.dollar_volume),
                liquid,
                "" if liquid else "minimum dollar volume not met",
            )
        )
        candidate_details[symbol] = details
        scored_by_symbol[symbol] = scored
    ranking = rank_candidates(candidates)
    audit.write(
        "MOMENTUM_RANKING",
        candle=candle,
        regime=regime.value,
        ranking=[
            {
                "rank": rank,
                "symbol": candidate.symbol,
                "momentum": candidate.momentum_score,
                "dollar_volume": candidate.dollar_volume,
            }
            for rank, candidate in enumerate(ranking, 1)
        ],
        liquidity_rejected=[candidate.symbol for candidate in candidates if not candidate.eligible],
    )
    new_positions = 0
    for candidate in ranking:
        symbol = candidate.symbol
        frame = frames[symbol]
        row = rows[symbol]
        scored = scored_by_symbol[symbol]
        details = candidate_details[symbol]
        if (
            scored.total < details["threshold"]
            or symbol in account.positions
            or symbol in state.positions
            or symbol in state.pending_orders
            or state.cooldowns.get(symbol, 0) > 0
            or sum(symbol in cfg.symbols for symbol in account.positions) + sum(
                side == "buy" and pending not in account.positions
                for pending, side in state.pending_orders.items()
            ) + (new_positions if settings.dry_run else 0) >= cfg.risk.max_positions
            or new_positions >= cfg.max_new_positions_per_cycle
        ):
            continue
        correlation_available = correlated_exposure_available(
            symbol, recent_returns, held_values, account.equity, cfg.risk
        )
        plan = size_position(
            account.equity,
            float(row.close),
            float(row.atr),
            min(available, correlation_available),
            cfg.risk,
        )
        if plan.quantity <= 0:
            audit.write("ENTRY_BLOCKED", symbol=symbol, reason="no exposure or risk capacity")
            continue
        client_id = deterministic_client_order_id(candle, symbol, "buy")
        if client_id in open_client_order_ids:
            state.pending_orders[symbol] = "buy"
            audit.write("DUPLICATE_ORDER_SUPPRESSED", symbol=symbol, side="buy", client_order_id=client_id)
            continue
        intent = OrderIntent(
            symbol, "buy", plan.quantity, "; ".join(scored.reasons), client_id
        )
        result = broker.submit(intent)
        submitted_order_id = order_id(result)
        audit.write(
            "BUY_SUBMITTED",
            **details,
            entry=float(row.close),
            atr=float(row.atr),
            initial_stop=plan.initial_stop,
            quantity=plan.quantity,
            notional=plan.notional,
            planned_risk=plan.risk_dollars,
            order_id=submitted_order_id,
            mode="dry_run" if settings.dry_run else "paper" if settings.paper_trading else "live",
        )
        if not settings.dry_run:
            state.positions[symbol] = ManagedPosition(
                symbol,
                float(row.close),
                plan.quantity,
                plan.initial_stop,
                float(row.close),
                now.isoformat(),
                scored.total,
                regime.value,
                scored.components,
                experiment_id,
                cfg.timeframe,
            )
            state.pending_orders[symbol] = "buy"
            if recorder:
                recorder.record_order(
                    {
                        "order_id": submitted_order_id,
                        "timestamp": now.isoformat(),
                        "candle": candle,
                        "symbol": symbol,
                        "side": "buy",
                        "reason": "; ".join(scored.reasons),
                        "client_order_id": client_id,
                        "quantity": plan.quantity,
                        "signal_price": float(row.close),
                        "submitted_at": now.isoformat(),
                        "experiment_id": experiment_id,
                        "timeframe": cfg.timeframe,
                        "regime": regime.value,
                        "score": scored.total,
                        "score_band": f"{int(scored.total // 10) * 10}-{int(scored.total // 10) * 10 + 9}",
                        **{f"component_{key}": value for key, value in scored.components.items()},
                    }
                )
        available -= plan.notional
        new_positions += 1
    store.save(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ccexchange crypto momentum bot")
    parser.add_argument("--once", action="store_true", help="Process one cycle and exit")
    parser.add_argument("--interval", type=int, default=None, help="Polling interval in seconds")
    parser.add_argument(
        "--paper-orders",
        action="store_true",
        help="Submit real orders to the Alpaca PAPER account; default is dry-run",
    )
    args = parser.parse_args()
    settings = RuntimeSettings()
    if args.paper_orders:
        if settings.live_trading or not settings.paper_trading:
            raise SystemExit("--paper-orders requires PAPER_TRADING=true and LIVE_TRADING=false")
        settings.dry_run = False
    settings.assert_execution_safe()
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env before running the bot")
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    audit = AuditLog()
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    cfg = load_config(settings.config_path)
    interval = args.interval or INTERVAL_SECONDS[cfg.timeframe]
    broker = make_broker(settings, cfg.backtest.initial_cash)
    store = StateStore()
    while not stopping:
        try:
            run_cycle(settings, audit, broker=broker, store=store)
        except Exception:
            LOGGER.exception("Bot cycle failed")
            if args.once:
                raise
        if args.once:
            return
        for _ in range(interval):
            if stopping:
                break
            time.sleep(1)
