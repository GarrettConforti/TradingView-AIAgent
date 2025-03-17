import numpy as np
np.NaN = np.nan  # Ensure uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests
from datetime import datetime

# -------------------------------
# 1. COINGECKO DATA FUNCTIONS
# -------------------------------

def parse_coingecko_coin_id(url: str):
    """
    Parse a CoinGecko URL (e.g. https://www.coingecko.com/en/coins/immutable-x)
    and return the coin ID (e.g. "immutable-x").
    """
    parts = url.strip().split('/')
    if 'coins' in parts:
        idx = parts.index('coins')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

def get_market_chart_data(coin_id: str, vs_currency='usd', days=1):
    """
    Fetch intraday market chart data from CoinGecko for the given coin_id.
    For days <= 1, CoinGecko returns data in 5-minute intervals.
    Returns a DataFrame with columns: ['price', 'volume'] and DateTime index.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        # Data fields: "prices": [[timestamp, price], ...], "total_volumes": [[timestamp, volume], ...]
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices or not volumes:
            return pd.DataFrame()
        df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
        df_vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df = pd.merge(df_prices, df_vol, on="timestamp")
        # Convert timestamp (ms) to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        st.write("Error fetching market chart data:", r.status_code)
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, frequency: str):
    """
    Resample the 5-minute data to the desired frequency (e.g., '15T' or '30T').
    Computes OHLC for 'price' and sum for 'volume'.
    Returns a DataFrame with columns: ['open', 'high', 'low', 'close', 'volume'].
    """
    if df.empty:
        return pd.DataFrame()
    ohlc = df["price"].resample(frequency).ohlc()
    vol = df["volume"].resample(frequency).sum()
    df_resampled = ohlc.join(vol)
    return df_resampled

def get_intraday_data(coin_id: str, vs_currency='usd', days=1, frequency='15T'):
    """
    Fetch intraday market chart data from CoinGecko and resample to the given frequency.
    """
    df_5m = get_market_chart_data(coin_id, vs_currency, days)
    if df_5m.empty:
        return pd.DataFrame()
    df_resampled = resample_ohlcv(df_5m, frequency)
    return df_resampled

# -------------------------------
# 2. TECHNICAL INDICATOR FUNCTIONS
# -------------------------------

def compute_indicators(df: pd.DataFrame):
    """
    Computes technical indicators using pandas_ta.
    Indicators: SMA(14), RSI(14), Bollinger Bands (20,2), MACD(12,26,9), Stochastic(14,3,3), ADX(14).
    """
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    df['STOCH_K'] = stoch['STOCHk_14_3_3']
    df['STOCH_D'] = stoch['STOCHd_14_3_3']
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    
    # Volume/Price action: Use price change and volume
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = 0
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 'vol_price_signal'] = 1
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), 'vol_price_signal'] = -1

    return df

def calculate_fibonacci_levels(df: pd.DataFrame, lookback=50):
    """
    Calculates Fibonacci retracement levels using the most recent 'lookback' bars.
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
# 3. WEIGHTED SCORING FUNCTIONS
# -------------------------------

def calculate_weighted_score(df: pd.DataFrame, fib_levels: dict):
    """
    Computes a weighted Prediction Score based on multiple indicator groups.
    Weights:
      - Momentum (RSI, Stochastic) ~ 25 points
      - Trend (SMA, MACD, ADX) ~ 35 points
      - Volatility (Bollinger) ~ 10 points
      - Volume/Price Action ~ 10 points
      - Fibonacci Retracement ~ 20 points
    """
    latest = df.iloc[-1]
    score = 0

    # Momentum (25 points)
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    if latest['STOCH_K'] < 20:
        score += 15
    elif latest['STOCH_K'] > 80:
        score -= 15

    # Trend (35 points)
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    if latest['MACD'] > latest['MACD_signal']:
        score += 15
    else:
        score -= 15

    if latest['ADX'] > 25:
        # Strong trend – add/subtract 10 based on MACD histogram
        if latest['MACD_hist'] > 0:
            score += 10
        else:
            score -= 10

    # Volatility (10 points)
    if latest['close'] < latest['BB_lower']:
        score += 10
    elif latest['close'] > latest['BB_upper']:
        score -= 10

    # Volume/Price Action (10 points)
    score += latest['vol_price_signal'] * 5

    # Fibonacci (20 points)
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 10
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 10

    return max(0, min(100, score))

def predict_trade_signal(score: float):
    """
    Converts the weighted score into a Buy/Sell/Neutral signal.
    """
    if score >= 70:
        return "Buy"
    elif score <= 30:
        return "Sell"
    else:
        return "Neutral"

def confidence_text(score: float):
    """
    Provides a text interpretation of swing-trade confidence based on the score.
    """
    if score >= 80:
        return "High confidence in a bullish swing trade."
    elif score >= 60:
        return "Moderate confidence in a bullish swing trade."
    elif score >= 40:
        return "Mixed signals; caution advised."
    else:
        return "Low confidence; market appears bearish."

# -------------------------------
# 4. ANALYSIS PIPELINE
# -------------------------------

def analyze_coin(coin_id: str, vs_currency='usd', days=1, frequency='15T', lookback=50):
    """
    Fetches intraday data from CoinGecko, resamples it, computes indicators,
    calculates a weighted Prediction Score, and returns the DataFrame and metrics.
    """
    df_raw = get_market_chart_data(coin_id, vs_currency, days)
    if df_raw.empty:
        return None, None, None, None, None
    df_resampled = resample_ohlcv(df_raw, frequency)
    if df_resampled.empty or len(df_resampled) < 30:
        return None, None, None, None, None
    df_resampled = compute_indicators(df_resampled)
    fib_levels = calculate_fibonacci_levels(df_resampled, lookback)
    score = calculate_weighted_score(df_resampled, fib_levels)
    signal = predict_trade_signal(score)
    conf = confidence_text(score)
    return df_resampled, score, signal, conf, fib_levels

# (Reusing earlier functions to fetch and resample 5m data)
def get_market_chart_data(coin_id: str, vs_currency='usd', days=1):
    """
    Fetch market chart data from CoinGecko for the given coin_id.
    Returns a DataFrame with 5-minute interval data if days <=1.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices or not volumes:
            return pd.DataFrame()
        df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
        df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df = pd.merge(df_prices, df_volumes, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        st.write("Error fetching CoinGecko data:", r.status_code)
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, frequency: str):
    """
    Resample 5-minute data into OHLCV bars of the specified frequency.
    """
    if df.empty:
        return pd.DataFrame()
    ohlc = df["price"].resample(frequency).ohlc()
    vol = df["volume"].resample(frequency).sum()
    df_resampled = ohlc.join(vol)
    return df_resampled

# -------------------------------
# 5. STREAMLIT APP INTERFACE
# -------------------------------

def main():
    st.title("Swinger - AI Agent")
    st.write("Analyze a coin's intraday data from CoinGecko for swing trading on 15m and 30m timeframes.")
    
    cg_url = st.text_input("CoinGecko URL (e.g., https://www.coingecko.com/en/coins/immutable-x)", "")
    days = st.number_input("Days of Data (should be <=1 for intraday, 1 for best resolution)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    
    if st.button("Run TA Analysis"):
        coin_id = parse_coingecko_coin_id(cg_url)
        if not coin_id:
            st.write("Invalid CoinGecko URL. Please enter a URL like https://www.coingecko.com/en/coins/immutable-x")
            return
        
        st.subheader("15-Minute Chart Analysis")
        df15, score15, signal15, conf15, fib15 = analyze_coin(coin_id, 'usd', days, '15T', lookback=50)
        if df15 is None or df15.empty:
            st.write("Not enough 15m data. Try increasing days or check the coin.")
        else:
            st.write(f"**15m Prediction Score:** {score15}/100")
            st.write(f"**15m Signal:** {signal15}")
            st.write(f"**Swing Trade Confidence:** {conf15}")
            st.write("Sample 15m Data (first 5 rows):")
            st.write(df15.head())
            plot_chart(df15, coin_id, '15m')
        
        st.subheader("30-Minute Chart Analysis")
        df30, score30, signal30, conf30, fib30 = analyze_coin(coin_id, 'usd', days, '30T', lookback=50)
        if df30 is None or df30.empty:
            st.write("Not enough 30m data. Try increasing days or check the coin.")
        else:
            st.write(f"**30m Prediction Score:** {score30}/100")
            st.write(f"**30m Signal:** {signal30}")
            st.write(f"**Swing Trade Confidence:** {conf30}")
            st.write("Sample 30m Data (first 5 rows):")
            st.write(df30.head())
            plot_chart(df30, coin_id, '30m')

if __name__ == "__main__":
    main()
