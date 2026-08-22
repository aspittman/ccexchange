# ccexchange

`ccexchange` is an independent, long-only crypto momentum research bot. Its question is deliberately falsifiable: does MomentumMaster's trend-following philosophy transfer from stocks to a volatile, continuously traded crypto market? It does not import or modify MomentumMaster, and it does not assume the answer is yes.

The initial allowlist is only `BTC/USD` and `ETH/USD`. New assets must be deliberately added to configuration and must pass a dollar-volume floor; there is no discovery or obscure-token scanner.

The starting universe is defined in `src/ccexchange/universe.py` and mirrored in
`config/default.yaml`. Each completed candle, the bot first rejects assets below
`liquidity.minimum_dollar_volume`, then ranks the survivors by momentum score. By default it may
open only the highest-ranked new opportunity per cycle (`max_new_positions_per_cycle: 1`). This
means ETH can receive capital ahead of BTC when its confirmed momentum is stronger. Future assets
such as SOL or LINK must be deliberately added to the configured allowlist; a high score can never
override the liquidity gate.

## Safety first

The defaults are both paper trading and dry-run:

```text
PAPER_TRADING=true
LIVE_TRADING=false
DRY_RUN=true
```

`ALPACA_PAPER=true` is accepted as an alias for `PAPER_TRADING=true`. Paper endpoint selection and
order authorization are intentionally separate: the paper flag chooses Alpaca's paper account,
while dry-run prevents submission until `--paper-orders` is supplied.

Dry-run computes orders without submitting them. Paper submission requires `DRY_RUN=false` while leaving paper enabled. Live submission is rejected unless paper and dry-run are both false, `LIVE_TRADING=true`, and `LIVE_TRADING_ACKNOWLEDGEMENT=I_UNDERSTAND_LIVE_CRYPTO_ORDERS` is supplied exactly. Credentials only come from environment variables. Start with a dedicated Alpaca paper account and never reuse live credentials during research.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# Only if .env does not already exist:
cp .env.example .env
ccexchange safety-check
pytest
```

Add your Alpaca paper credentials to the two blank fields in `.env`. Trading remains paper-first and dry-run because those are application defaults.

## Run the bot

Run one signal-calculation cycle without submitting orders:

```bash
python main.py --once
```

Run continuously in the foreground:

```bash
python main.py
```

Use the supervisor for unattended operation. It restarts `main.py` after an unexpected non-zero exit and forwards Ctrl-C/termination cleanly:

```bash
python launcher.py
```

After reviewing dry-run logs, explicitly enable Alpaca paper order submission with:

```bash
python3 main.py --paper-orders
# or supervised
python3 launcher.py --paper-orders
```

This flag cannot enable live trading. The process reconciles broker positions, persists stop,
high-watermark, pending-order, and circuit-breaker state under `state/`, and suppresses duplicate
orders. Existing positions continue to receive exit management while new entries are suspended.
ATR stops are managed by the bot process, so they depend on this process and Alpaca being available.

During `--paper-orders` operation, the bot also maintains a paper-research dataset under
`paper_data/`: deduplicated completed OHLCV bars, account-equity snapshots, reconciled Alpaca fills,
round trips, and `paper_report.json`. Refresh or print the report at any time with:

```bash
ccexchange paper-report
```

The report includes total return, drawdown, volatility, Sharpe ratio, fill and round-trip counts,
win rate, average winner/loser, and profit factor. Historical backtests remain separate so paper
results are never confused with simulated results.

Every paper configuration receives a stable experiment ID and a complete JSON snapshot under
`paper_data/experiments/`. `decisions.csv` retains eligible and rejected signals with raw indicator
values and score components; `orders.csv` links those conditions to bot-tagged Alpaca fills.
`technique_analysis.json` and `.csv` summarize outcomes by symbol, regime, timeframe, score band,
configuration, indicator confirmation, and exit reason. Groups with fewer than 20 completed trades
are marked as insufficient samples. These comparisons are descriptive rather than causal; validate
parameter changes with walk-forward testing.

Collected bars are directly reusable by the historical engine. For the default daily timeframe:

```bash
ccexchange backtest --data paper_data/bars/1Day --output results/paper-period-replay.json
```

Launcher activity is written to `logs/launcher.log`; strategy, regime, score, and calculated-order events are written to `logs/events.jsonl`. The entire project already lives in this base directory—there is no second nested project folder. The `src/ccexchange` directory is only the importable Python package.

## Strategy

Completed candles feed EMA structure, MACD level and acceleration, ADX, Bollinger breakout and Band Width expansion, normalized volume, and relative strength. ETH is compared with BTC; BTC receives its own market/trend test. Components contribute configurable points rather than acting as mandatory binary gates. Every signal event includes its component ledger and reasons.

BTC drives one of five regimes: strong uptrend, uptrend, sideways/choppy, downtrend, or high-risk/crash. Sideways and downtrend regimes raise the entry hurdle; high-risk blocks entries. ATR determines position risk and the initial/ratcheting stop. A stop can rise with the high-water mark but cannot loosen when volatility expands. Emergency loss, daily portfolio loss, drawdown, per-position, and total-exposure limits are independently configurable.

The bot exits on a ratcheting stop, emergency loss, high-risk regime, or combined MACD/EMA deterioration. It does not sell merely because price touched the upper band or became “overbought.”

## Data and backtesting

Place UTC OHLCV CSVs at `data/BTC_USD.csv` and `data/ETH_USD.csv`, with columns:

```text
timestamp,open,high,low,close,volume
```

Run:

```bash
ccexchange backtest --config config/default.yaml --data data --output results/daily.json
```

Signals are calculated after a completed bar and filled at the next bar's open with configurable fees and adverse slippage. Results include total return, CAGR, drawdown, volatility, Sharpe, Sortino, win rate, average winners/losers, profit factor, trade count, turnover, and invested/cash time. They are shown beside BTC, ETH, and equal-weight passive benchmarks. Open positions are marked to the final close.

For 4-hour or 12-hour research, use only fully completed bars. Alpaca supplies native hourly bars; aggregate them with right-closed/right-labeled windows before passing them to the engine. `1Day`, `4Hour`, `12Hour`, and `1Hour` annualization are supported. Daily and 4-hour are the intended first experiments.

## Walk-forward experiments

`ccexchange.walkforward.walk_forward` accepts a small, explicit list of parameter patches. For each fold it evaluates candidates only on the training window, selects by training Sharpe, then runs that selection once on the following untouched window. It retains every in-sample candidate and the out-of-sample outcome. Use a hypothesis-led grid (for example three ADX thresholds and three ATR multipliers), not hundreds of combinations. Compare fold behavior across known bull, bear, crash, sideways, recovery, and volatility environments and disclose weak results.

## Architecture

The package separates `data`, `indicators`, `scoring`, `regime`, `relative_strength`, `risk`, `portfolio`, `execution`, `audit`, `backtest`, `walkforward`, and `config`. Alpaca appears only behind data and broker adapters; the research engine has no brokerage dependency.

This software is experimental, not investment advice. Paper results can materially overstate executable performance in 24/7 crypto markets.
