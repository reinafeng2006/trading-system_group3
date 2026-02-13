"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from sentiment_analyzer import MediaSentimentAnalyzer

class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df

class SentimentMomentumStrategy(Strategy):
    """
    Momentum strategy enhanced with media sentiment analysis.
    
    Technical Signal:
    - Momentum: pct_change over lookback
    - Volatility: rolling std of returns
    - Momentum confidence: momentum / volatility
    
    Sentiment Signal:
    - Fetches news from Finnhub + Google News
    - Analyzes with FinBERT
    - Returns confidence [0, 1] where 0.5 is neutral
    
    Combined Signal:
    - combined_conf = momentum_conf * (2 * sentiment_conf - 1)
    - This scales momentum_conf by sentiment in range [-1, +1]
    
    Position Sizing:
    - Scales with combined confidence
    - Long if combined_conf > threshold
    - Short if combined_conf < -threshold
    """
    
    def __init__(
        self,
        lookback: int = 20,
        vol_window: int = 30,
        conf_threshold: float = 0.6,
        position_size: float = 10.0,
        max_scale: float = 3.0,
        sentiment_weight: float = 0.5,  # How much to weight sentiment (0-1)
        sentiment_cache_hours: int = 6,  # Cache sentiment for N hours
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
        self.sentiment_cache_hours = sentiment_cache_hours
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = MediaSentimentAnalyzer()
        
        # Cache for sentiment scores
        self._sentiment_cache = {}  # symbol -> (timestamp, confidence)
    
    def get_sentiment_confidence(self, symbol: str) -> float:
        """
        Get sentiment confidence for a symbol.
        Uses cache to avoid excessive API calls.
        
        Returns:
            Confidence in [0, 1] where 0.5 is neutral
        """
        now = datetime.now()
        
        # Check cache
        if symbol in self._sentiment_cache:
            cached_time, cached_conf = self._sentiment_cache[symbol]
            age_hours = (now - cached_time).total_seconds() / 3600
            
            if age_hours < self.sentiment_cache_hours:
                print(f"Using cached sentiment for {symbol}: {cached_conf:.4f} (age: {age_hours:.1f}h)")
                return cached_conf
        
        # Fetch new sentiment
        try:
            print(f"Fetching fresh sentiment for {symbol}...")
            confidence, results = self.sentiment_analyzer.analyze_symbol(
                symbol, 
                hours_back=24
            )
            
            # Cache the result
            self._sentiment_cache[symbol] = (now, confidence)
            
            print(f"Sentiment confidence for {symbol}: {confidence:.4f}")
            return confidence
            
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            print("Using neutral sentiment (0.5)")
            return 0.5  # Neutral if sentiment fetch fails
    
    def add_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Add momentum indicators and fetch sentiment.
        
        Args:
            df: Price data
            symbol: Stock ticker (required for sentiment)
        """
        df = df.copy()
        
        # Technical indicators (same as original)
        df["ret"] = df["Close"].pct_change().fillna(0.0)
        df["mom"] = df["Close"].pct_change(self.lookback)
        df["vol"] = df["ret"].rolling(self.vol_window, min_periods=2).std()
        
        # Momentum confidence ratio
        vol_floor = df["vol"].replace(0.0, np.nan)
        df["momentum_conf"] = (df["mom"] / vol_floor).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        
        # Fetch sentiment confidence (single value for entire dataframe)
        if symbol:
            sentiment_conf = self.get_sentiment_confidence(symbol)
        else:
            print("Warning: No symbol provided, using neutral sentiment")
            sentiment_conf = 0.5
        
        # Store sentiment confidence as column
        df["sentiment_conf"] = sentiment_conf
        
        # Convert sentiment from [0, 1] to [-1, +1] (0.5 neutral -> 0)
        df["sentiment_normalized"] = 2 * df["sentiment_conf"] - 1
        
        # Combined confidence: momentum scaled by sentiment
        # If sentiment is bullish (>0.5), amplify positive momentum
        # If sentiment is bearish (<0.5), amplify negative momentum
        df["combined_conf"] = df["momentum_conf"] * (
            1 + self.sentiment_weight * df["sentiment_normalized"]
        )
        
        # Scaling for position sizing
        df["scale"] = (df["combined_conf"].abs() / self.conf_threshold).clip(
            0.0, self.max_scale
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on combined confidence."""
        df = df.copy()
        
        # Initialize signal
        df["signal"] = 0
        
        # Long signal: combined confidence > threshold
        df.loc[df["combined_conf"] > self.conf_threshold, "signal"] = 1
        
        # Short signal: combined confidence < -threshold
        df.loc[df["combined_conf"] < -self.conf_threshold, "signal"] = -1
        
        # Position matches signal
        df["position"] = df["signal"]
        
        # Size scales with combined confidence
        df["target_qty"] = df["position"].abs() * self.position_size * (1.0 + df["scale"])
        
        return df
    
    def run(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Execute the full strategy pipeline.
        
        Args:
            df: Price data
            symbol: Stock ticker (required for sentiment)
        """
        df = df.copy()
        df = self.add_indicators(df, symbol=symbol)
        df = self.generate_signals(df)
        return df
