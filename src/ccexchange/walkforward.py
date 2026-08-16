from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import Backtester, BacktestResult
from .config import StrategyConfig


@dataclass
class Fold:
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    training_results: list[dict]
    selected_parameters: dict
    out_of_sample: BacktestResult


def walk_forward(
    data: dict[str, pd.DataFrame],
    base: StrategyConfig,
    experiments: list[dict],
    train_bars: int,
    test_bars: int,
) -> list[Fold]:
    """Optimize only inside each training window; report the following untouched test window."""
    index = data[base.benchmark].index
    folds = []
    for start in range(0, len(index) - train_bars - test_bars + 1, test_bars):
        train_idx = index[start : start + train_bars]
        test_idx = index[start + train_bars : start + train_bars + test_bars]
        candidates = []
        for patch in experiments:
            cfg = base.model_copy(deep=True)
            for section, values in patch.items():
                target = getattr(cfg, section)
                for key, value in values.items():
                    setattr(target, key, value)
            result = Backtester(cfg).run({s: f.loc[train_idx] for s, f in data.items()})
            candidates.append((result.metrics["sharpe_ratio"], patch, result.metrics))
        _, selected, _ = max(candidates, key=lambda x: x[0])
        cfg = base.model_copy(deep=True)
        for section, values in selected.items():
            for key, value in values.items():
                setattr(getattr(cfg, section), key, value)
        warmup = max(
            base.minimum_history_bars,
            base.indicators.ema_long,
            base.indicators.bandwidth_lookback + base.indicators.bollinger_period,
        )
        warm_start = max(0, start + train_bars - warmup)
        combined_idx = index[warm_start : start + train_bars + test_bars]
        oos = Backtester(cfg).run(
            {s: f.loc[combined_idx] for s, f in data.items()}, start_at=test_idx[0]
        )
        folds.append(
            Fold(
                train_idx[0],
                train_idx[-1],
                test_idx[0],
                test_idx[-1],
                [{"parameters": p, "metrics": m} for _, p, m in candidates],
                selected,
                oos,
            )
        )
    return folds
