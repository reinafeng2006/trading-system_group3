"""
Alpaca paper-trading runner.

Requires environment variables (loaded from .env):
    ALPACA_API_KEY
    ALPACA_API_SECRET
    ALPACA_API_URL (optional, defaults to paper endpoint)
    ALPACA_DATA_FEED (optional, defaults to iex for stocks)

Usage:
    python run_live.py --symbol AAPL --asset-class stock --strategy ma --timeframe 1Min
"""

from __future__ import annotations

import argparse
import time

from core.alpaca_trader import AlpacaTrader
from pipeline.alpaca import clean_market_data, save_bars
from strategies import MovingAverageStrategy, TemplateStrategy, get_strategy_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-trading loop with Alpaca.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker or crypto symbol.")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe (e.g., 1Min, 5Min).")
    parser.add_argument("--lookback", type=int, default=200, help="Bars to fetch each iteration.")
    parser.add_argument("--strategy", default="ma", help="Strategy name (ma, template, or a class name).")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (MA strategy).")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (MA strategy).")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size.")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback (template).")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold (template).")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold (template).")
    parser.add_argument("--iterations", type=int, default=1, help="How many loops to run.")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between loops.")
    parser.add_argument("--live", action="store_true", help="Run forever until stopped.")
    parser.add_argument("--save-data", action="store_true", help="Save raw+clean CSVs to data/.")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without placing orders.")
    parser.add_argument("--feed", default=None, help="Data feed (iex or sip for stocks).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy_cls = get_strategy_class(args.strategy)
    if strategy_cls is MovingAverageStrategy:
        strategy = MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    elif strategy_cls is TemplateStrategy:
        strategy = TemplateStrategy(
            lookback=args.momentum_lookback,
            position_size=args.position_size,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    else:
        try:
            strategy = strategy_cls()
        except TypeError as exc:
            raise SystemExit(
                f"{strategy_cls.__name__} must support a no-arg constructor or use --strategy template."
            ) from exc

    trader = AlpacaTrader(
        symbol=args.symbol,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
        lookback=args.lookback,
        strategy=strategy,
        feed=args.feed,
        dry_run=args.dry_run,
    )

    def handle_iteration() -> None:
        df = trader.run_once()
        if args.save_data and df is not None:
            raw_path = save_bars(df, args.symbol, args.timeframe, args.asset_class)
            clean_market_data(raw_path)

    if args.live:
        try:
            while True:
                handle_iteration()
                time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("Stopped.")
    else:
        for i in range(args.iterations):
            handle_iteration()
            if i < args.iterations - 1:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
