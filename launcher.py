#!/usr/bin/env python3
"""Supervise main.py and restart it after unexpected failures."""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "launcher.log"
LOGGER = logging.getLogger("ccexchange.launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the ccexchange bot running")
    parser.add_argument("--once", action="store_true", help="Run one bot cycle and exit")
    parser.add_argument(
        "--paper-orders",
        action="store_true",
        help="Forward explicit Alpaca paper-order authorization to main.py",
    )
    parser.add_argument("--restart-delay", type=float, default=5.0)
    parser.add_argument("--max-restart-delay", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
    )
    stopping = False
    child: subprocess.Popen | None = None

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        LOGGER.info("Shutdown requested")
        if child and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    delay = max(0.0, args.restart_delay)
    while not stopping:
        command = [sys.executable, str(PROJECT_ROOT / "main.py")]
        if args.once:
            command.append("--once")
        if args.paper_orders:
            command.append("--paper-orders")
        started = time.monotonic()
        LOGGER.info("Starting bot: %s", " ".join(command))
        child = subprocess.Popen(command, cwd=PROJECT_ROOT)
        code = child.wait()
        runtime = time.monotonic() - started
        child = None
        if stopping or code == 0 or args.once:
            LOGGER.info("Bot exited with status %s", code)
            return code
        LOGGER.error("Bot crashed with status %s; restarting in %.1f seconds", code, delay)
        time.sleep(delay)
        delay = (
            args.restart_delay if runtime >= 300 else min(max(delay * 2, 1), args.max_restart_delay)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
