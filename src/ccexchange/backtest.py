from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .indicators import add_indicators
from .portfolio import Portfolio, Position
from .regime import MarketRegime, classify_regime, required_entry_score
from .relative_strength import relative_strength
from .risk import CircuitBreaker, ratchet_stop, size_position
from .scoring import momentum_score
from .universe import RankedCandidate, rank_candidates


@dataclass
class Trade:
    symbol: str
    entry_time: object
    exit_time: object
    entry: float
    exit: float
    quantity: float
    pnl: float
    return_pct: float
    reason: str
    mfe: float
    mae: float
    entry_score: float
    regime: str
    components: dict[str, float]


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    trades: list[Trade]
    equity: pd.Series
    events: list[dict]
    benchmarks: dict[str, dict[str, float]]


def performance(
    equity: pd.Series, trades: list[Trade], periods_per_year: float
) -> dict[str, float]:
    returns = equity.pct_change().fillna(0)
    years = max(len(equity) / periods_per_year, 1 / periods_per_year)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * math.sqrt(periods_per_year)
    downside = returns.clip(upper=0).std() * math.sqrt(periods_per_year)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    invested = float(returns.index.to_series().map(lambda _: True)) if False else 0.0
    return {
        "total_return": total,
        "cagr": cagr,
        "maximum_drawdown": float((equity / equity.cummax() - 1).min()),
        "volatility": vol,
        "sharpe_ratio": (returns.mean() * periods_per_year / vol if vol else 0),
        "sortino_ratio": (returns.mean() * periods_per_year / downside if downside else 0),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "average_winner": float(np.mean(wins)) if wins else 0,
        "average_loser": float(np.mean(losses)) if losses else 0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0),
        "number_of_trades": len(trades),
        "turnover": sum(abs(t.quantity * t.entry) + abs(t.quantity * t.exit) for t in trades)
        / equity.iloc[0],
        "time_invested": invested,
        "time_in_cash": 1 - invested,
    }


def passive(close: pd.Series, initial: float, periods: float) -> dict[str, float]:
    curve = initial * close / close.iloc[0]
    return performance(curve, [], periods)


class Backtester:
    """Long-only, completed-bar signals executed at the next candle open."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def run(self, raw: dict[str, pd.DataFrame], start_at=None) -> BacktestResult:
        periods = {"1Day": 365, "4Hour": 2190, "12Hour": 730, "1Hour": 8760}.get(
            self.cfg.timeframe, 365
        )
        frames = {s: add_indicators(f, self.cfg.indicators, periods) for s, f in raw.items()}
        benchmark = frames[self.cfg.benchmark]
        for symbol, frame in frames.items():
            frame["relative_strength"] = (
                0.0
                if symbol == self.cfg.benchmark
                else relative_strength(
                    frame.close, benchmark.close, self.cfg.indicators.relative_strength_period
                )
            )
            frame["bb_width_prev"] = frame.bb_width.shift()
        index = benchmark.index.intersection(
            pd.Index(sorted(set.intersection(*(set(f.index) for f in frames.values()))))
        )
        if start_at is not None:
            index = index[index >= pd.Timestamp(start_at)]
        if len(index) < 2:
            raise ValueError("backtest evaluation window must contain at least two aligned bars")
        portfolio = Portfolio(self.cfg.backtest.initial_cash)
        breaker = CircuitBreaker(portfolio.cash, portfolio.cash, portfolio.cash)
        pending: list[dict] = []
        trades: list[Trade] = []
        events: list[dict] = []
        curve = {}
        invested_count = 0
        initial = portfolio.cash
        for n, ts in enumerate(index):
            prices = {s: float(f.loc[ts].close) for s, f in frames.items()}
            # Execute prior completed-candle decisions at this candle's open.
            for order in pending:
                s, side = order["symbol"], order["side"]
                row = frames[s].loc[ts]
                px = float(row.open) * (
                    1 + (1 if side == "buy" else -1) * self.cfg.backtest.slippage_bps / 10000
                )
                fee = px * order["quantity"] * self.cfg.backtest.fee_bps / 10000
                if side == "buy" and px * order["quantity"] + fee <= portfolio.cash:
                    portfolio.cash -= px * order["quantity"] + fee
                    portfolio.positions[s] = Position(
                        s, order["quantity"], px, order["stop"], px, ts, entry_fee=fee
                    )
                    events.append({**order, "event": "BUY", "time": ts, "price": px})
                elif side == "sell" and s in portfolio.positions:
                    p = portfolio.positions.pop(s)
                    proceeds = px * p.quantity - fee
                    portfolio.cash += proceeds
                    pnl = proceeds - p.entry * p.quantity - p.entry_fee
                    trades.append(
                        Trade(
                            s,
                            p.entry_time,
                            ts,
                            p.entry,
                            px,
                            p.quantity,
                            pnl,
                            px / p.entry - 1,
                            order["reason"],
                            p.mfe,
                            p.mae,
                            order["entry_score"],
                            order["regime"],
                            order["components"],
                        )
                    )
                    events.append({**order, "event": "SELL", "time": ts, "price": px, "pnl": pnl})
            pending = []
            equity = portfolio.equity(prices)
            curve[ts] = equity
            invested_count += bool(portfolio.positions)
            risk_reason = breaker.reason(equity, self.cfg.risk)
            btc = benchmark.loc[ts]
            if pd.isna(btc.ema_long) or n + 1 >= len(index):
                continue
            regime, regime_score = classify_regime(btc, self.cfg)
            for symbol, p in list(portfolio.positions.items()):
                row = frames[symbol].loc[ts]
                # The stop entering this bar may fill intrabar. A newly ratcheted stop only
                # becomes active on the next bar, avoiding unknowable high/low ordering bias.
                if float(row.low) <= p.stop:
                    raw_fill = min(p.stop, float(row.open))
                    px = raw_fill * (1 - self.cfg.backtest.slippage_bps / 10000)
                    fee = px * p.quantity * self.cfg.backtest.fee_bps / 10000
                    portfolio.cash += px * p.quantity - fee
                    portfolio.positions.pop(symbol)
                    pnl = px * p.quantity - fee - p.entry * p.quantity - p.entry_fee
                    trades.append(
                        Trade(
                            symbol,
                            p.entry_time,
                            ts,
                            p.entry,
                            px,
                            p.quantity,
                            pnl,
                            px / p.entry - 1,
                            "ratcheting ATR stop",
                            p.mfe,
                            p.mae,
                            0,
                            regime.value,
                            {},
                        )
                    )
                    events.append(
                        {
                            "event": "SELL",
                            "symbol": symbol,
                            "time": ts,
                            "price": px,
                            "reason": "ratcheting ATR stop",
                            "pnl": pnl,
                        }
                    )
                    continue
                p.high = max(p.high, float(row.high))
                p.mfe = max(p.mfe, float(row.high / p.entry - 1))
                p.mae = min(p.mae, float(row.low / p.entry - 1))
                p.stop = ratchet_stop(p.stop, p.high, float(row.atr), self.cfg.risk)
                reason = None
                if row.close <= p.entry * (1 - self.cfg.risk.emergency_stop_percent):
                    reason = "emergency stop"
                elif regime == MarketRegime.HIGH_RISK:
                    reason = "market regime deterioration"
                elif row.macd < row.macd_signal and row.close < row.ema_fast:
                    reason = "MACD and EMA momentum deterioration"
                if reason:
                    pending.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "quantity": p.quantity,
                            "reason": reason,
                            "entry_score": 0,
                            "regime": regime.value,
                            "components": {},
                        }
                    )
            if risk_reason:
                events.append(
                    {"event": "RISK", "time": ts, "reason": risk_reason, "equity": equity}
                )
                continue
            exposure_available = max(
                0, equity * self.cfg.risk.max_crypto_exposure - portfolio.exposure(prices)
            )
            candidates = []
            scores = {}
            for symbol, frame in frames.items():
                row = frame.loc[ts]
                score = momentum_score(
                    row, self.cfg, float(row.relative_strength), symbol == self.cfg.benchmark
                )
                liquid = bool(row.dollar_volume >= self.cfg.liquidity.minimum_dollar_volume)
                candidates.append(
                    RankedCandidate(symbol, score.total, float(row.dollar_volume), liquid)
                )
                scores[symbol] = score
                threshold = required_entry_score(regime, self.cfg)
                events.append(
                    {
                        "event": "SIGNAL",
                        "time": ts,
                        "symbol": symbol,
                        "regime": regime.value,
                        "regime_score": regime_score,
                        "score": score.total,
                        "threshold": threshold,
                        "components": score.components,
                        "reasons": score.reasons,
                        "liquidity_eligible": liquid,
                    }
                )
            ranking = rank_candidates(candidates)
            events.append(
                {
                    "event": "MOMENTUM_RANKING",
                    "time": ts,
                    "ranking": [
                        {
                            "rank": rank,
                            "symbol": item.symbol,
                            "momentum": item.momentum_score,
                            "dollar_volume": item.dollar_volume,
                        }
                        for rank, item in enumerate(ranking, 1)
                    ],
                }
            )
            new_positions = 0
            threshold = required_entry_score(regime, self.cfg)
            for candidate in ranking:
                symbol = candidate.symbol
                if (
                    symbol in portfolio.positions
                    or any(o["symbol"] == symbol for o in pending)
                    or new_positions >= self.cfg.max_new_positions_per_cycle
                ):
                    continue
                row = frames[symbol].loc[ts]
                score = scores[symbol]
                if score.total >= threshold:
                    plan = size_position(
                        equity, float(row.close), float(row.atr), exposure_available, self.cfg.risk
                    )
                    if plan.quantity > 0:
                        pending.append(
                            {
                                "symbol": symbol,
                                "side": "buy",
                                "quantity": plan.quantity,
                                "stop": plan.initial_stop,
                                "reason": "; ".join(score.reasons),
                                "entry_score": score.total,
                                "regime": regime.value,
                                "components": score.components,
                            }
                        )
                        exposure_available -= plan.notional
                        new_positions += 1
        # Mark open positions to final close so results do not silently omit them.
        final_prices = {s: float(f.loc[index[-1]].close) for s, f in frames.items()}
        final_equity = portfolio.equity(final_prices)
        curve[index[-1]] = final_equity
        eq = pd.Series(curve, dtype=float)
        metrics = performance(eq, trades, periods)
        metrics["time_invested"] = invested_count / len(eq) if len(eq) else 0
        metrics["time_in_cash"] = 1 - metrics["time_invested"]
        benchmarks = {s: passive(f.loc[index].close, initial, periods) for s, f in frames.items()}
        if len(frames) >= 2:
            blend = (
                sum(f.loc[index].close / f.loc[index].close.iloc[0] for f in frames.values())
                / len(frames)
                * initial
            )
            benchmarks["equal_weight"] = performance(blend, [], periods)
        return BacktestResult(metrics, trades, eq, events, benchmarks)


def result_dict(result: BacktestResult) -> dict:
    return {
        "metrics": result.metrics,
        "benchmarks": result.benchmarks,
        "trades": [asdict(t) for t in result.trades],
        "events": result.events,
    }
