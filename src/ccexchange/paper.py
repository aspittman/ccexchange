from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .execution import AccountSnapshot, FilledOrder, OrderUpdate


class PaperRecorder:
    """Persist paper-market inputs, account equity, fills, and reproducible reports."""

    def __init__(self, root: str | Path = "paper_data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_bars(self, frames: dict[str, pd.DataFrame], timeframe: str) -> None:
        bars_dir = self.root / "bars" / timeframe
        bars_dir.mkdir(parents=True, exist_ok=True)
        for symbol, frame in frames.items():
            path = bars_dir / f"{symbol.replace('/', '_')}.csv"
            incoming = frame.reset_index().rename(
                columns={frame.index.name or "index": "timestamp"}
            )
            if path.exists():
                incoming = pd.concat([pd.read_csv(path), incoming], ignore_index=True)
            # CSV timestamps are loaded as strings, while fresh bar indexes are
            # usually pandas Timestamps.  Normalize both before sorting and also
            # collapse equivalent timestamps written with different offsets.
            incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
            incoming.drop_duplicates("timestamp", keep="last").sort_values("timestamp").to_csv(
                path, index=False
            )

    def record_snapshot(self, timestamp: datetime, account: AccountSnapshot) -> None:
        path = self.root / "equity.csv"
        row = pd.DataFrame(
            [
                {
                    "timestamp": timestamp.isoformat(),
                    "equity": account.equity,
                    "cash": account.cash,
                    "crypto_exposure": sum(p.market_value for p in account.positions.values()),
                }
            ]
        )
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([existing, row], ignore_index=True).drop_duplicates(
            "timestamp", keep="last"
        ).to_csv(path, index=False)

    def record_fills(self, fills: list[FilledOrder]) -> None:
        if not fills:
            return
        path = self.root / "fills.csv"
        incoming = pd.DataFrame([asdict(fill) for fill in fills])
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([existing, incoming], ignore_index=True).drop_duplicates(
            "order_id", keep="last"
        ).sort_values("filled_at").to_csv(path, index=False)

    def record_order_updates(self, updates: list[OrderUpdate]) -> None:
        if not updates:
            return
        path = self.root / "order_updates.csv"
        incoming = pd.DataFrame([asdict(update) for update in updates])
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([existing, incoming], ignore_index=True).drop_duplicates(
            "order_id", keep="last"
        ).to_csv(path, index=False)

    def register_experiment(self, config: dict) -> str:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        experiment_id = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        directory = self.root / "experiments"
        directory.mkdir(exist_ok=True)
        path = directory / f"{experiment_id}.json"
        if not path.exists():
            path.write_text(
                json.dumps({"experiment_id": experiment_id, "config": config}, indent=2),
                encoding="utf-8",
            )
        return experiment_id

    def record_decision(self, decision: dict) -> None:
        self._append_row("decisions.csv", decision, ["experiment_id", "candle", "symbol"])

    def record_order(self, order: dict) -> None:
        self._append_row("orders.csv", order, ["order_id"])

    def _append_row(self, filename: str, row: dict, keys: list[str]) -> None:
        path = self.root / filename
        incoming = pd.DataFrame([row])
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([existing, incoming], ignore_index=True).drop_duplicates(
            keys, keep="last"
        ).to_csv(path, index=False)

    def report(self) -> dict:
        equity_path, fills_path = self.root / "equity.csv", self.root / "fills.csv"
        metrics = {
            "total_return": 0.0,
            "cagr": 0.0,
            "maximum_drawdown": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "number_of_fills": 0,
            "number_of_round_trips": 0,
            "win_rate": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "profit_factor": 0.0,
            "turnover": 0.0,
            "time_invested": 0.0,
            "time_in_cash": 1.0,
            "fill_rate": 0.0,
            "average_slippage_bps": 0.0,
            "average_fill_latency_seconds": 0.0,
            "rejected_orders": 0,
            "canceled_orders": 0,
            "partially_filled_orders": 0,
        }
        if equity_path.exists():
            equity = pd.read_csv(equity_path, parse_dates=["timestamp"]).sort_values("timestamp")
            series = equity.set_index("timestamp").equity.groupby(level=0).last()
            if len(series) >= 2:
                daily = series.resample("1D").last().ffill()
                returns = daily.pct_change().dropna()
                metrics["total_return"] = float(series.iloc[-1] / series.iloc[0] - 1)
                elapsed_years = (series.index[-1] - series.index[0]).total_seconds() / (
                    365.25 * 86400
                )
                if elapsed_years >= 1 / 365.25 and series.iloc[-1] > 0:
                    metrics["cagr"] = float(
                        (series.iloc[-1] / series.iloc[0]) ** (1 / elapsed_years) - 1
                    )
                metrics["maximum_drawdown"] = float((series / series.cummax() - 1).min())
                metrics["volatility"] = float(returns.std() * np.sqrt(365)) if len(returns) else 0.0
                metrics["sharpe_ratio"] = (
                    float(returns.mean() / returns.std() * np.sqrt(365))
                    if len(returns) > 1 and returns.std()
                    else 0.0
                )
                downside = returns[returns < 0].std()
                metrics["sortino_ratio"] = (
                    float(returns.mean() / downside * np.sqrt(365))
                    if downside and not np.isnan(downside)
                    else 0.0
                )
            invested = equity.crypto_exposure > 0
            metrics["time_invested"] = float(invested.mean())
            metrics["time_in_cash"] = 1 - metrics["time_invested"]
        round_trips = self._round_trips(fills_path)
        fill_frame = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
        metrics["number_of_fills"] = len(fill_frame)
        if not fill_frame.empty and equity_path.exists():
            starting_equity = pd.read_csv(equity_path).equity.iloc[0]
            metrics["turnover"] = float(
                (fill_frame.quantity * fill_frame.price).abs().sum() / starting_equity
            )
        metrics["number_of_round_trips"] = len(round_trips)
        metrics.update(self._execution_quality(fill_frame))
        if round_trips:
            pnls = [trade["pnl"] for trade in round_trips]
            wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
            metrics["win_rate"] = len(wins) / len(pnls)
            metrics["average_winner"] = float(np.mean(wins)) if wins else 0.0
            metrics["average_loser"] = float(np.mean(losses)) if losses else 0.0
            metrics["profit_factor"] = (
                sum(wins) / abs(sum(losses)) if losses else float("inf") if wins else 0.0
            )
        report = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "metrics": metrics,
            "round_trips": round_trips,
            "note": "Paper fills exclude fees unless reported in execution price.",
        }
        report["technique_analysis"] = self._technique_analysis(round_trips)
        (self.root / "paper_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame(round_trips).to_csv(self.root / "round_trips.csv", index=False)
        return report

    def _execution_quality(self, fills: pd.DataFrame) -> dict[str, float]:
        orders_path = self.root / "orders.csv"
        empty = {
            "fill_rate": 0.0,
            "average_slippage_bps": 0.0,
            "average_fill_latency_seconds": 0.0,
            "rejected_orders": 0,
            "canceled_orders": 0,
            "partially_filled_orders": 0,
        }
        if not orders_path.exists():
            return empty
        orders = pd.read_csv(orders_path)
        if orders.empty:
            return empty
        filled_ids = set(fills.order_id.astype(str)) if not fills.empty else set()
        result = {
            **empty,
            "fill_rate": len(set(orders.order_id.astype(str)) & filled_ids) / len(orders),
        }
        updates_path = self.root / "order_updates.csv"
        if updates_path.exists():
            updates = pd.read_csv(updates_path)
            statuses = updates.status.astype(str).str.lower()
            result["rejected_orders"] = int(statuses.eq("rejected").sum())
            result["canceled_orders"] = int(
                statuses.isin({"canceled", "expired", "done_for_day"}).sum()
            )
            requested = updates.requested_quantity.astype(float)
            filled = updates.filled_quantity.astype(float)
            result["partially_filled_orders"] = int(
                ((filled > 0) & (filled < requested)).sum()
            )
        required = {"order_id", "signal_price", "submitted_at"}
        if fills.empty or not required.issubset(orders.columns):
            return result
        joined = orders.merge(fills, on="order_id", suffixes=("_order", "_fill"))
        joined = joined.dropna(subset=["signal_price", "price", "submitted_at", "filled_at"])
        if joined.empty:
            return result
        direction = joined["side_order"].map({"buy": 1.0, "sell": -1.0})
        slippage = direction * (joined.price.astype(float) / joined.signal_price.astype(float) - 1)
        latency = (
            pd.to_datetime(joined.filled_at, utc=True)
            - pd.to_datetime(joined.submitted_at, utc=True)
        ).dt.total_seconds()
        result["average_slippage_bps"] = float(slippage.mean() * 10_000)
        result["average_fill_latency_seconds"] = float(latency.mean())
        return result

    @staticmethod
    def _round_trips(path: Path) -> list[dict]:
        if not path.exists():
            return []
        fills = pd.read_csv(path).sort_values("filled_at")
        inventory: dict[str, list[dict]] = {}
        trades = []
        for fill in fills.itertuples():
            lots = inventory.setdefault(fill.symbol, [])
            if fill.side == "buy":
                lots.append(
                    {
                        "quantity": float(fill.quantity),
                        "price": float(fill.price),
                        "time": fill.filled_at,
                        "order_id": fill.order_id,
                    }
                )
            elif fill.side == "sell":
                remaining = float(fill.quantity)
                while remaining > 1e-12 and lots:
                    lot = lots[0]
                    quantity = min(remaining, lot["quantity"])
                    pnl = quantity * (float(fill.price) - lot["price"])
                    trades.append(
                        {
                            "symbol": fill.symbol,
                            "entry_time": lot["time"],
                            "exit_time": fill.filled_at,
                            "quantity": quantity,
                            "entry": lot["price"],
                            "exit": float(fill.price),
                            "pnl": pnl,
                            "return_pct": float(fill.price) / lot["price"] - 1,
                            "entry_order_id": lot["order_id"],
                            "exit_order_id": fill.order_id,
                        }
                    )
                    remaining -= quantity
                    lot["quantity"] -= quantity
                    if lot["quantity"] <= 1e-12:
                        lots.pop(0)
        return trades

    def _technique_analysis(self, round_trips: list[dict]) -> dict:
        orders_path = self.root / "orders.csv"
        if not round_trips or not orders_path.exists():
            return {
                "sample_size": 0,
                "warning": "No completed attributed trades yet.",
                "groups": [],
            }
        trades = pd.DataFrame(round_trips)
        orders = pd.read_csv(orders_path)
        entries = orders[orders.side == "buy"].add_prefix("entry_")
        joined = trades.merge(
            entries, left_on="entry_order_id", right_on="entry_order_id", how="left"
        )
        exits = orders[orders.side == "sell"].add_prefix("exit_")
        joined = joined.merge(exits, left_on="exit_order_id", right_on="exit_order_id", how="left")
        groups = []
        dimensions = [
            "symbol",
            "entry_regime",
            "entry_timeframe",
            "entry_score_band",
            "entry_experiment_id",
            "exit_reason",
        ]
        dimensions += [column for column in joined if column.startswith("entry_component_")]
        for dimension in dimensions:
            if dimension not in joined:
                continue
            for value, sample in joined.groupby(dimension, dropna=False):
                returns = sample.return_pct.astype(float)
                groups.append(
                    {
                        "dimension": dimension.removeprefix("entry_"),
                        "value": str(value),
                        "trades": len(sample),
                        "win_rate": float((returns > 0).mean()),
                        "average_return": float(returns.mean()),
                        "total_pnl": float(sample.pnl.sum()),
                        "insufficient_sample": len(sample) < 20,
                    }
                )
        pd.DataFrame(groups).to_csv(self.root / "technique_analysis.csv", index=False)
        result = {
            "sample_size": len(joined),
            "warning": "Associations are descriptive, not causal; prefer 20+ trades per group and walk-forward validation.",
            "groups": groups,
        }
        (self.root / "technique_analysis.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
