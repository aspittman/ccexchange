import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import Backtester, result_dict
from .config import RuntimeSettings, load_config


def main():
    parser = argparse.ArgumentParser(prog="ccexchange")
    parser.add_argument("command", choices=["safety-check", "backtest"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="results/backtest.json")
    args = parser.parse_args()
    runtime = RuntimeSettings()
    runtime.assert_execution_safe()
    if args.command == "safety-check":
        print(
            json.dumps(
                {
                    "paper_trading": runtime.paper_trading,
                    "live_trading": runtime.live_trading,
                    "dry_run": runtime.dry_run,
                    "safe": True,
                }
            )
        )
        return
    cfg = load_config(args.config or runtime.config_path)
    raw = {}
    for symbol in cfg.symbols:
        path = Path(args.data) / f"{symbol.replace('/', '_')}.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        raw[symbol] = frame[["open", "high", "low", "close", "volume"]]
    result = Backtester(cfg).run(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_dict(result), default=str, indent=2), encoding="utf-8")
    result.equity.rename("equity").to_csv(output.with_suffix(".equity.csv"))
    print(json.dumps({"metrics": result.metrics, "benchmarks": result.benchmarks}, indent=2))


if __name__ == "__main__":
    main()
