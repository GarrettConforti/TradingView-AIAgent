import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import ccxt
import math

# -------------------------------
# 1. DATA FETCH & INDICATORS
# -------------------------------

def get_data(symbol, exchange_name, timeframe='15m', limit=200):
    """
    Fetches OHLCV data from a specified exchange (e.g., kraken, coinbasepro, gemini)
    for a given symbol and timeframe.
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.write(f"Error fetching data for {symbol} on {exchange_name} ({timeframe}):", e)
        return pd.DataFrame()

def compute_advanced_indicators(df):
    """
    Computes an extended set of technical indicators:
      - RSI (14)
      - SMA (14)
      - Bollinger Bands (20,2)
      - MACD (12,26,9)
      - Stochastic (14,3,3)
      - ADX (14)
    """
    # RSI, SMA, Bollinger from your original approach
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['SMA14'] = ta.sma(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']

    # Stochastic (14,3,3)
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    df['STOCH_K'] = stoch['STOCHk_14_3_3']
    df['STOCH_D'] = stoch['STOCHd_14_3_3']

    # ADX (14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']

    # Volume Price Action: We'll replicate your approach
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = 0
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 'vol_price_signal'] = 1
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), 'vol_price_signal'] = -1

    return df

def calculate_fibonacci_levels(df, lookback=100):
    """
    Calculate Fibonacci retracement using the last `lookback` bars.
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

# -------------------------------
# 2. WEIGHTED SCORING LOGIC
# -------------------------------

def calculate_advanced_percentage_wheel(df, fib_levels):
    """
    Uses multiple indicators to form a weighted score (0-100).
    We group indicators by category, each with its own weight:
      - Momentum (RSI, Stochastic) ~ 25 points
      - Trend (SMA, MACD, ADX) ~ 35 points
      - Volatility (Bollinger) ~ 10 points
      - Volume/Price action ~ 10 points
      - Fibonacci retracement ~ 20 points

    Adjust these weights or logic to suit your preference.
    """
    latest = df.iloc[-1]
    score = 0

    # ========== Momentum Indicators (25 points max) ==========
    # RSI (10 points)
    if latest['RSI'] < 30:
        score += 10  # strongly bullish
    elif latest['RSI'] > 70:
        # negative contribution
        # if it's above 70, we subtract up to 10 points
        score -= 10

    # Stochastic (K & D) (15 points)
    # If STOCH_K < 20 => bullish, STOCH_K > 80 => bearish
    # We'll do a simple approach
    stoch_k = latest['STOCH_K']
    if stoch_k < 20:
        score += 15
    elif stoch_k > 80:
        score -= 15

    # ========== Trend Indicators (35 points max) ==========
    # SMA14 (10 points)
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    # MACD (15 points)
    # If MACD > signal => bullish, else => bearish
    if latest['MACD'] > latest['MACD_signal']:
        score += 15
    else:
        score -= 15

    # ADX (10 points)
    # If ADX > 25 => trending market => if other signals are bullish, add; if not, subtract
    adx_val = latest['ADX']
    if adx_val > 25:
        # We'll add or subtract based on overall direction
        # If MACD hist > 0 => bullish
        if latest['MACD_hist'] > 0:
            score += 10
        else:
            score -= 10
    else:
        # no strong trend
        pass

    # ========== Volatility (10 points max) ==========
    # Bollinger Bands
    if latest['close'] < latest['BB_lower']:
        # oversold
        score += 10
    elif latest['close'] > latest['BB_upper']:
        # overbought
        score -= 10

    # ========== Volume/Price Action (10 points max) ==========
    # from vol_price_signal
    # 1 => +5, -1 => -5
    # We'll do a simple approach
    volume_signal = latest['vol_price_signal']
    score += (volume_signal * 5)

    # ========== Fibonacci Retracement (20 points max) ==========
    # If near 38.2 or 50 => +10 each for bullish
    # If near 23.6 => -10
    close_val = latest['close']
    tolerance = 0.01 * close_val
    if abs(close_val - fib_levels['38.2%']) < tolerance or abs(close_val - fib_levels['50%']) < tolerance:
        score += 10
    if abs(close_val - fib_levels['23.6%']) < tolerance:
        score -= 10

    # Bound the score between 0 and 100
    score = max(0, min(100, score))
    return score

def predict_signal(score):
    """
    Convert final weighted score into Buy/Sell/Neutral.
    """
    if score >= 70:
        return "Buy"
    elif score <= 30:
        return "Sell"
    else:
        return "Neutral"

def confidence_text(score):
    """
    Provide a short text about swing-trade confidence based on the final score.
    """
    if score >= 80:
        return "High confidence in a bullish swing trade."
    elif score >= 60:
        return "Moderate confidence in a bullish swing trade."
    elif score >= 40:
        return "Neutral to slight bearish; swing trade is risky."
    else:
        return "Likely bearish conditions; low confidence in a buy swing trade."

# -------------------------------
# 3. PLOTTING & MAIN LOGIC
# -------------------------------

def analyze_timeframe(symbol, exchange_name, timeframe='15m', lookback=100):
    """
    Fetch data, compute advanced indicators, fib levels, weighted score, and produce final signals.
    """
    df = get_data(symbol, exchange_name, timeframe, limit=200)
    if df.empty or len(df) < 30:
        return None, None, None, None, None

    df = compute_advanced_indicators(df)
    fib_levels = calculate_fibonacci_levels(df, lookback)
    score = calculate_advanced_percentage_wheel(df, fib_levels)
    signal = predict_signal(score)
    conf_text = confidence_text(score)
    return df, score, signal, conf_text, fib_levels

def plot_chart(df, symbol, timeframe):
    """
    Plot close price with some key indicators for a quick visual.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{symbol} - {timeframe} Analysis")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 4. STREAMLIT APP
# -------------------------------

def main():
    st.title("Advanced Swing-Trade Analysis (15m & 30m) with Weighted Indicators")
    st.write("We incorporate RSI, SMA, Bollinger, MACD, Stochastic, ADX, Volume/Price, and Fibonacci for a more detailed score.")

    # Input fields
    symbol = st.text_input("Trading Pair (e.g., BTC/USD)", value="BTC/USD")
    exchange_name = st.selectbox("Select Exchange", ["kraken", "coinbasepro", "gemini"])
    lookback = st.number_input("Fibonacci Lookback (bars)", min_value=20, max_value=200, value=100)

    if st.button("Analyze Swing Trade"):
        # 15m
        st.subheader("15-Minute Analysis")
        df15, score15, signal15, conf15, fib15 = analyze_timeframe(symbol, exchange_name, '15m', lookback)
        if df15 is None:
            st.write("Not enough data for 15m timeframe.")
        else:
            st.write(f"**15m Score:** {score15} / 100")
            st.write(f"**15m Signal:** {signal15}")
            st.write(f"**Swing-Trade Confidence:** {conf15}")
            st.write("First 5 rows of data:")
            st.write(df15.head())
            plot_chart(df15, symbol, '15m')

        # 30m
        st.subheader("30-Minute Analysis")
        df30, score30, signal30, conf30, fib30 = analyze_timeframe(symbol, exchange_name, '30m', lookback)
        if df30 is None:
            st.write("Not enough data for 30m timeframe.")
        else:
            st.write(f"**30m Score:** {score30} / 100")
            st.write(f"**30m Signal:** {signal30}")
            st.write(f"**Swing-Trade Confidence:** {conf30}")
            st.write("First 5 rows of data:")
            st.write(df30.head())
            plot_chart(df30, symbol, '30m')

if __name__ == "__main__":
    main()
