"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Inherit from Strategy
2. Implement add_indicators(df)
3. Implement generate_signals(df)

Required output columns from generate_signals():
    signal: 1=buy, -1=sell, 0=hold
    target_qty: position size
    position: 1=long, -1=short, 0=flat

Optional:
    limit_price: places limit order instead of market
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentiment_analyzer import MediaSentimentAnalyzer


class Strategy:
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        try:
            df = self.add_indicators(df, **kwargs)
        except TypeError:
            df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class SentimentMomentumStrategy(Strategy):
    """
    Momentum strategy enhanced with media sentiment analysis.

    Technical Signal:
        Momentum: pct_change over lookback
        Volatility: rolling std of returns
        Momentum confidence: momentum / volatility

    Sentiment Signal:
        Fetches news from Finnhub + Google News relative to the bar's timestamp.
        Analyzes with FinBERT. Returns confidence [0, 1] where 0.5 is neutral.

    Combined Signal:
        combined_conf = momentum_conf * (1 + sentiment_weight * sentiment_normalized)

    Position Sizing:
        Scales with combined confidence.
        Long if combined_conf > threshold, short if < -threshold.
    """

    def __init__(
        self,
        lookback: int = 120,
        vol_window: int = 30,
        conf_threshold: float = 1.6,
        position_size: float = 10.0,
        max_scale: float = 3.0,
        sentiment_weight: float = 0.5,
        sentiment_cache_minutes: float = 5.0,
    ):
        if lookback < 1 or vol_window < 2:
            raise ValueError("lookback must be >= 1 and vol_window must be >= 2")
        if position_size <= 0:
            raise ValueError("position_size must be positive")
        if not 0 <= sentiment_weight <= 1:
            raise ValueError("sentiment_weight must be between 0 and 1")

        self.lookback = lookback
        self.vol_window = vol_window
        self.conf_threshold = conf_threshold
        self.position_size = position_size
        self.max_scale = max_scale
        self.sentiment_weight = sentiment_weight
        self.sentiment_cache_minutes = sentiment_cache_minutes

        self.sentiment_analyzer = MediaSentimentAnalyzer()
        self._sentiment_cache = {}  # symbol -> (cached_reference_time, confidence)

    def get_sentiment_confidence(self, symbol: str, reference_time: datetime = None) -> float:
        if reference_time is None:
            reference_time = datetime.now()
        reference_time = reference_time.replace(tzinfo=None)

        if symbol in self._sentiment_cache:
            cached_time, cached_conf = self._sentiment_cache[symbol]
            age_minutes = (reference_time - cached_time).total_seconds() / 60
            if age_minutes < self.sentiment_cache_minutes:
                print(f"Using cached sentiment for {symbol}: {cached_conf:.4f} (age: {age_minutes:.1f}m)")
                return cached_conf

        try:
            print(f"Fetching fresh sentiment for {symbol} at {reference_time}...")
            confidence, _ = self.sentiment_analyzer.analyze_symbol(
                symbol,
                hours_back=24,
                reference_time=reference_time
            )
            self._sentiment_cache[symbol] = (reference_time, confidence)
            print(f"Sentiment confidence for {symbol}: {confidence:.4f}")
            return confidence

        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            print("Using neutral sentiment (0.5)")
            return 0.5

    def add_indicators(self, df: pd.DataFrame, symbol: str = None, reference_time: datetime = None) -> pd.DataFrame:
        df = df.copy()

        df["ret"] = df["Close"].pct_change().fillna(0.0)
        df["mom"] = df["Close"].pct_change(self.lookback)
        df["vol"] = df["ret"].rolling(self.vol_window, min_periods=2).std()

        vol_floor = df["vol"].replace(0.0, np.nan)
        df["momentum_conf"] = (df["mom"] / vol_floor).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        if symbol:
            sentiment_conf = self.get_sentiment_confidence(symbol, reference_time=reference_time)
        else:
            print("Warning: No symbol provided, using neutral sentiment")
            sentiment_conf = 0.5

        df["sentiment_conf"] = sentiment_conf
        df["sentiment_normalized"] = 2 * df["sentiment_conf"] - 1
        df["combined_conf"] = df["momentum_conf"] * (
            1 + self.sentiment_weight * df["sentiment_normalized"]
        )
        df["combined_conf"] = df["combined_conf"].fillna(0)
        df["scale"] = (df["combined_conf"].abs() / self.conf_threshold).clip(0.0, self.max_scale)
        df["scale"] = df["scale"].fillna(0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        go_long = (df["combined_conf"].shift(1) <= self.conf_threshold) & (
            df["combined_conf"] > self.conf_threshold)
        go_short = (df["combined_conf"].shift(1) >= -self.conf_threshold) & (
            df["combined_conf"] < -self.conf_threshold)

        df.loc[go_long, "signal"] = 1
        df.loc[go_short, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = (df["position"].abs() * self.position_size * (
            1.0 + df["scale"]) / (1.0 + df["vol"] * 15))
        df = df.fillna(0)
        return df

    def run(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        reference_time = pd.Timestamp(df.iloc[-1]["Datetime"]).to_pydatetime().replace(tzinfo=None)
        df = self.add_indicators(df, symbol=symbol, reference_time=reference_time)
        df = self.generate_signals(df)
        return df


def get_strategy_class(name: str):
    strategies = {
        "sentiment": SentimentMomentumStrategy,
    }
    if name in strategies:
        return strategies[name]
    raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
