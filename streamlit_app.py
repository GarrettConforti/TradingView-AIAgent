import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import ccxt
import requests

# -------------------------------
# 1. EXCHANGE DATA & TA FUNCTIONS
# -------------------------------

def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    """
    Fetches OHLCV data for a given trading pair from the specified US-based exchange.
    Example exchanges: kraken, coinbasepro, gemini
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.write(f"Data fetched from {exchange_name} for {timeframe} timeframe for {symbol}.")
    except Exception as e:
        st.write(f"Error fetching data for {symbol} in {timeframe} timeframe:", e)
        df = pd.DataFrame()
    return df

def compute_indicators(df):
    """
    Computes common technical indicators using pandas_ta.
    """
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    return df

def calculate_fibonacci_levels(df, lookback=100):
    """
    Calculates Fibonacci retracement levels using the most recent 'lookback' bars.
    """
    recent = df[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    levels = {
        '0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100%': low
    }
    return levels

def analyze_volume_price(df):
    """
    Computes a simple volume-price action signal.
    """
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

def calculate_percentage_wheel(df, fib_levels):
    """
    Aggregates various technical signals into a Prediction Score (0 to 100).
    """
    latest = df.iloc[-1]
    score = 50  # Neutral starting point

    # RSI: Oversold adds bullish bias; overbought adds bearish bias.
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    # SMA: Price above SMA indicates bullish trend.
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    # Bollinger Bands: Price near lower band suggests oversold conditions.
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    # Fibonacci retracement: Proximity to key levels can adjust bias.
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price Action: Supports the overall trend.
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

def predict_signal(score):
    """
    Converts the Prediction S
