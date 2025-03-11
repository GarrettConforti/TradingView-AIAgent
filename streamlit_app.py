import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests

# -------------------------------
# 1. COINGECKO TA FUNCTIONS
# -------------------------------

def parse_coingecko_coin_id(url: str):
    """
    Example: https://www.coingecko.com/en/coins/immutable-x
    We'll parse out 'immutable-x' as the coin_id.
    """
    parts = url.strip().split('/')
    # We look for the segment after 'coins'
    if 'coins' in parts:
        idx = parts.index('coins')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

def get_coingecko_ohlc(coin_id: str, vs_currency='usd', days=7):
    """
    Fetch daily OHLC data from CoinGecko for the given coin_id over `days`.
    Endpoint: /coins/{id}/ohlc?vs_currency=usd&days=7
    Returns a list of [timestamp, open, high, low, close].
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": vs_currency,
        "days": days
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if data and isinstance(data, list):
            return data
    return None

def build_dataframe_from_ohlc(ohlc_data):
    """
    Convert the list of [timestamp, open, high, low, close] into a Pandas DataFrame
    with columns: ['open', 'high', 'low', 'close'] and index = datetime in ms.
    CoinGecko returns timestamp in milliseconds for daily data.
    """
    df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    # Convert timestamp (ms) to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # CoinGecko's daily OHLC doesn't include volume. We'll set volume to 0 for now.
    df['volume'] = 0

    return df

def compute_indicators(df: pd.DataFrame):
    """
    Computes SMA(14), RSI(14), and Bollinger Bands (period=20, std=2).
    """
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    return df

def analyze_volume_price(df: pd.DataFrame):
    """
    Creates a dummy volume/price signal. Since we have no real volume from CoinGecko daily,
    we'll set volume=0. This means 'vol_price_signal' will always be 0.
    """
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = 0  # Because volume=0, there's no real volume-based signal
    return df

def calculate_fibonacci_levels(df: pd.DataFrame, lookback=100):
    """
    Calculate Fibonacci retracement using the last `lookback` bars (daily).
    """
    recent = df.tail(lookback)
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

def calculate_percentage_wheel(df: pd.DataFrame, fib_levels: dict):
    """
    Aggregates signals into a Prediction Score (0 to 100).
    """
    latest = df.iloc[-1]
    score = 50  # Neutral starting point

    # RSI
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    # SMA
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    # Bollinger Bands
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    # Fibonacci Retracement
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price Action
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

def predict_signal(score: float):
    """
    Convert Prediction Score to Buy/Sell/Neutral.
    """
    if score > 55:
        return "Buy"
    elif score < 45:
        return "Sell"
    else:
        return "Neutral"

def explain_signal(df: pd.DataFrame, score: float, signal: str, fib_levels: dict):
    """
    Provide a short textual explanation based on the indicators.
    """
    latest = df.iloc[-1]
    rsi_val = latest['RSI']
    sma_val = latest['SMA14']

    if signal == "Buy":
        return (f"RSI at {rsi_val:.1f} suggests oversold conditions, price is above SMA(14) at {sma_val:.2f}, "
                "and price is near key Fibonacci levels. Overall signals point to a bullish trend.")
    elif signal == "Sell":
        return (f"RSI at {rsi_val:.1f} indicates overbought conditions, price is below SMA(14) at {sma_val:.2f}, "
                "and price is away from support levels. Signals suggest a bearish trend.")
    else:
        return "Indicators are mixed or neutral, no strong directional bias."

def coingecko_ta_analysis(coin_id: str, vs_currency='usd', days=7):
    """
    1) Fetch daily OHLC from CoinGecko for the last `days`.
    2) Build DataFrame, compute indicators, compute Prediction Score.
    3) Return (df, score, signal, explanation).
    """
    ohlc_data = get_coingecko_ohlc(coin_id, vs_currency, days)
    if not ohlc_data:
        return None, None, None, None

    df = build_dataframe_from_ohlc(ohlc_data)
    df = compute_indicators(df)
    df = analyze_volume_price(df)
    fib_levels = calculate_fibonacci_levels(df, lookback=min(days, len(df)))
    score = calculate_percentage_wheel(df, fib_levels)
    signal = predict_signal(score)
    explanation = explain_signal(df, score, signal, fib_levels)

    return df, score, signal, explanation

def plot_daily_chart(df: pd.DataFrame, coin_id: str):
    """
    Plots daily close + TA indicators for the coin.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Close Price (Daily)')
    ax.plot(df['SMA14'], label='SMA14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{coin_id} - Daily Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 2. STREAMLIT APP
# -------------------------------

def main():
    st.title("CoinGecko TA Analysis (Daily)")
    st.write("Fetch daily OHLC data from CoinGecko and run TA indicators (SMA, RSI, Bollinger, Fibonacci).")
    st.write("Because CoinGecko only provides daily OHLC, this is effectively a '1-Day timeframe' analysis.")

    # User inputs
    cg_url = st.text_input("CoinGecko URL (e.g., https://www.coingecko.com/en/coins/immutable-x)", "")
    days = st.number_input("Days of Daily Data to Fetch", min_value=1, max_value=90, value=7, step=1)

    if st.button("Run TA Analysis"):
        # 1) Parse coin_id from the URL
        coin_id = parse_coingecko_coin_id(cg_url)
        if not coin_id:
            st.write("Could not parse a valid coin ID from the provided URL.")
            return

        # 2) Perform TA
        df, score, signal, explanation = coingecko_ta_analysis(coin_id, 'usd', days)
        if df is None or df.empty:
            st.write("No OHLC data found for that coin/timeframe on CoinGecko.")
        else:
            # Show first few rows
            st.write("Daily OHLC Data (first 5 rows):")
            st.write(df.head())

            st.write(f"**Daily Prediction Score:** {score}%")
            st.write(f"**Daily Trade Signal:** {signal}")
            st.write(f"**Reasoning:** {explanation}")

            # Plot daily chart
            plot_daily_chart(df, coin_id)

if __name__ == "__main__":
    main()
