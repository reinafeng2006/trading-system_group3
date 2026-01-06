from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
from alpaca_trade_api.rest import APIError

from pipeline.alpaca import fetch_crypto_bars, fetch_stock_bars, get_rest
from strategies import Strategy


@dataclass
class TradeDecision:
    side: str
    qty: float
    price: float
    order_type: str
    limit_price: Optional[float] = None


class AlpacaTrader:
    """
    Simple paper-trading loop that uses Alpaca for data and orders.
    """

    def __init__(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        lookback: int,
        strategy: Strategy,
        feed: Optional[str] = None,
        dry_run: bool = False,
        api: Optional[tradeapi.REST] = None,
    ):
        asset_class = asset_class.lower()
        if asset_class not in {"stock", "crypto"}:
            raise ValueError("asset_class must be 'stock' or 'crypto'.")

        self.symbol = symbol.upper()
        self.asset_class = asset_class
        self.timeframe = timeframe
        self.lookback = lookback
        self.strategy = strategy
        self.feed = feed
        self.dry_run = dry_run
        self.api = api or get_rest()
        self.starting_equity = self._get_equity()

    def _get_equity(self) -> float:
        account = self.api.get_account()
        return float(account.equity)

    def _get_net_position(self) -> float:
        try:
            position = self.api.get_position(self.symbol)
        except APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                return 0.0
            raise

        qty = float(position.qty)
        if getattr(position, "side", "long") == "short":
            qty = -qty
        return qty

    def _has_open_order(self) -> bool:
        orders = self.api.list_orders(status="open", symbols=[self.symbol])
        return len(orders) > 0

    def fetch_latest_bars(self) -> pd.DataFrame:
        if self.asset_class == "crypto":
            return fetch_crypto_bars(
                self.symbol,
                timeframe=self.timeframe,
                limit=self.lookback,
                api=self.api,
            )
        return fetch_stock_bars(
            self.symbol,
            timeframe=self.timeframe,
            limit=self.lookback,
            feed=self.feed,
            api=self.api,
        )

    def _format_qty(self, qty: float) -> str:
        if self.asset_class == "crypto":
            return f"{qty:.6f}".rstrip("0").rstrip(".")
        return str(int(qty))

    def _build_decision(self, df: pd.DataFrame) -> Optional[TradeDecision]:
        if df.empty:
            return None

        signals_df = self.strategy.run(df)
        latest = signals_df.iloc[-1]
        signal_value = latest.get("signal", 0)
        signal = int(signal_value) if pd.notna(signal_value) else 0
        if signal == 0:
            return None

        qty_value = latest.get("target_qty", 0)
        qty = float(qty_value) if pd.notna(qty_value) else 0.0
        if qty <= 0:
            return None

        limit_value = latest.get("limit_price", None)
        limit_price = float(limit_value) if pd.notna(limit_value) else None
        order_type = "limit" if limit_price is not None else "market"

        side = "buy" if signal > 0 else "sell"
        price = float(latest.get("Close", 0.0))
        return TradeDecision(side=side, qty=qty, price=price, order_type=order_type, limit_price=limit_price)

    def _adjust_qty_for_position(self, decision: TradeDecision, net_position: float) -> float:
        qty = decision.qty
        if decision.side == "buy":
            if net_position > 0:
                return 0.0
            if net_position < 0:
                qty = abs(net_position) + qty
        else:
            if net_position < 0:
                return 0.0
            if net_position > 0:
                qty = abs(net_position) + qty

        if self.asset_class == "stock":
            qty = float(int(qty))
            if qty <= 0:
                return 0.0
        return qty

    def _submit_order(self, decision: TradeDecision, qty: float) -> Optional[str]:
        tif = "gtc" if self.asset_class == "crypto" else "day"
        order_kwargs = {"type": decision.order_type, "time_in_force": tif}
        if decision.order_type == "limit":
            order_kwargs["limit_price"] = decision.limit_price

        if self.dry_run:
            return "dry_run"

        qty_to_send = int(qty) if self.asset_class == "stock" else qty
        order = self.api.submit_order(
            symbol=self.symbol,
            qty=qty_to_send,
            side=decision.side,
            **order_kwargs,
        )
        return order.id

    def _print_trade(self, decision: TradeDecision, qty: float, order_id: str) -> None:
        equity = self._get_equity()
        net_pnl = equity - self.starting_equity
        qty_display = self._format_qty(qty)
        print_price = decision.limit_price if decision.order_type == "limit" else decision.price
        price_display = f"{print_price:.2f}" if print_price else "market"
        timestamp = pd.Timestamp.utcnow()
        print(
            f"{timestamp:%Y-%m-%d %H:%M:%S} | {decision.side.upper()} {qty_display} {self.symbol} @ {price_display} "
            f"| order_id={order_id} | net_pnl={net_pnl:+.2f}"
        )

    def run_once(self) -> Optional[pd.DataFrame]:
        try:
            df = self.fetch_latest_bars()
        except ValueError as exc:
            print(str(exc))
            return None
        decision = self._build_decision(df)
        if decision is None:
            return df

        if self._has_open_order():
            return df

        net_position = self._get_net_position()
        qty = self._adjust_qty_for_position(decision, net_position)
        if qty <= 0:
            return df

        order_id = self._submit_order(decision, qty)
        self._print_trade(decision, qty, order_id or "unknown")
        return df

    def run(self, iterations: int = 1, sleep_seconds: int = 60) -> None:
        for i in range(iterations):
            self.run_once()
            if i < iterations - 1:
                time.sleep(sleep_seconds)
