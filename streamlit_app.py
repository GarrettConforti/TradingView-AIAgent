import numpy as np
np.NaN = np.nan  # Ensure uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests

######################################
# 1. HELPER FUNCTIONS (CoinGecko + Resampling)
######################################

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
    Fetch market chart data from CoinGecko for the given coin_id.
    For days <=1, it returns ~5-minute intervals.
    Returns a DataFrame with columns: ['price', 'volume'] and a DateTime index.
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
    Resample 5-minute data into OHLCV bars for the specified frequency (e.g. '15T', '30T').
    Returns a DataFrame with columns: ['open', 'high', 'low', 'close', 'volume'].
    """
    if df.empty:
        return pd.DataFrame()
    # OHLC on 'price'
    ohlc = df["price"].resample(frequency).ohlc()
    # Sum 'volume'
    vol = df["volume"].resample(frequency).sum()
    df_resampled = ohlc.join(vol)
    return df_resampled

######################################
# 2. TECHNICAL INDICATORS + WEIGHTED SCORING
######################################

def compute_indicators(df: pd.DataFrame):
    """
    Computes advanced indicators using pandas_ta:
      - SMA(14), RSI(14), Bollinger(20,2)
      - MACD(12,26,9), Stochastic(14,3,3), ADX(14)
      - Volume/Price signal
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
    
    # Volume/Price action
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = 0
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 'vol_price_signal'] = 1
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), 'vol_price_signal'] = -1
    
    return df

def calculate_fibonacci_levels(df: pd.DataFrame, lookback=50):
    """
    Calculates Fibonacci retracement levels on the last `lookback` bars.
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

def calculate_weighted_score(df: pd.DataFrame, fib_levels: dict):
    """
    Uses multiple indicators to form a weighted score (0-100).
      - Momentum (RSI, Stoch) ~25 points
      - Trend (SMA, MACD, ADX) ~35 points
      - Volatility (Bollinger) ~10 points
      - Volume/Price ~10 points
      - Fibonacci ~20 points
    """
    latest = df.iloc[-1]
    score = 0
    
    # Momentum
    # RSI
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10
    # Stoch
    if latest['STOCH_K'] < 20:
        score += 15
    elif latest['STOCH_K'] > 80:
        score -= 15
    
    # Trend
    # SMA14
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10
    # MACD
    if latest['MACD'] > latest['MACD_signal']:
        score += 15
    else:
        score -= 15
    # ADX
    if latest['ADX'] > 25:
        if latest['MACD_hist'] > 0:
            score += 10
        else:
            score -= 10
    
    # Volatility (Bollinger)
    if latest['close'] < latest['BB_lower']:
        score += 10
    elif latest['close'] > latest['BB_upper']:
        score -= 10
    
    # Volume/Price
    score += latest['vol_price_signal'] * 5  # max +5 or -5
    
    # Fibonacci
    close_val = latest['close']
    tolerance = 0.01 * close_val
    if abs(close_val - fib_levels['38.2%']) < tolerance or abs(close_val - fib_levels['50%']) < tolerance:
        score += 10
    if abs(close_val - fib_levels['23.6%']) < tolerance:
        score -= 10
    
    return max(0, min(100, score))

def predict_trade_signal(score: float):
    """
    Convert the weighted score into a Buy/Sell/Neutral signal.
    """
    if score >= 70:
        return "Buy"
    elif score <= 30:
        return "Sell"
    else:
        return "Neutral"

def confidence_text(score: float):
    """
    Provide a text interpretation of swing-trade confidence based on the score.
    """
    if score >= 80:
        return "High confidence in a bullish swing trade."
    elif score >= 60:
        return "Moderate confidence in a bullish swing trade."
    elif score >= 40:
        return "Mixed signals; caution advised."
    else:
        return "Low confidence; market appears bearish."

######################################
# 3. CHART PLOTTING FUNCTION
######################################

def plot_chart(df: pd.DataFrame, coin_id: str, timeframe: str):
    """
    Plot the close price with some key indicators for quick visual analysis.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{coin_id} - {timeframe} Chart Analysis")
    ax.legend()
    st.pyplot(fig)

######################################
# 4. ANALYSIS PIPELINE
######################################

def analyze_coin(coin_id: str, vs_currency='usd', days=1, frequency='15T', lookback=50):
    """
    1) Fetch ~5-minute data from CoinGecko for `days`.
    2) Resample to the desired frequency (15m, 30m).
    3) Compute advanced indicators, weighted score, trade signal, and confidence text.
    4) Return DataFrame, score, signal, confidence, fib_levels.
    """
    df_raw = get_market_chart_data(coin_id, vs_currency, days)
    if df_raw.empty:
        return None, None, None, None, None
    
    df_resampled = resample_ohlcv(df_raw, frequency)
    if df_resampled.empty or len(df_resampled) < 10:
        return None, None, None, None, None
    
    df_resampled = compute_indicators(df_resampled)
    fib_levels = calculate_fibonacci_levels(df_resampled, lookback)
    score = calculate_weighted_score(df_resampled, fib_levels)
    signal = predict_trade_signal(score)
    conf = confidence_text(score)
    
    return df_resampled, score, signal, conf, fib_levels

######################################
# 5. STREAMLIT APP
######################################

def main():
    st.title("Swinger - AI Agent")
    st.write("Analyze a coin's intraday data from CoinGecko for swing trading on 15m and 30m timeframes.")
    
    cg_url = st.text_input("CoinGecko URL (e.g., https://www.coingecko.com/en/coins/immutable-x)", "")
    days = st.number_input("Days of Data (<=1 for best 5m intervals)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    
    if st.button("Run TA Analysis"):
        coin_id = parse_coingecko_coin_id(cg_url)
        if not coin_id:
            st.write("Invalid CoinGecko URL. Must be like https://www.coingecko.com/en/coins/immutable-x")
            return
        
        # 15-Minute Analysis
        st.subheader("15-Minute Chart Analysis")
        df15, score15, signal15, conf15, fib15 = analyze_coin(coin_id, 'usd', days, '15T', lookback=50)
        if df15 is None or df15.empty:
            st.write("Not enough 15m data. Try increasing 'days' or check the coin.")
        else:
            st.write(f"**15m Prediction Score:** {score15}/100")
            st.write(f"**15m Signal:** {signal15}")
            st.write(f"**Swing-Trade Confidence:** {conf15}")
            st.write("Sample 15m Data (first 5 rows):")
            st.write(df15.head())
            plot_chart(df15, coin_id, '15m')
        
        # 30-Minute Analysis
        st.subheader("30-Minute Chart Analysis")
        df30, score30, signal30, conf30, fib30 = analyze_coin(coin_id, 'usd', days, '30T', lookback=50)
        if df30 is None or df30.empty:
            st.write("Not enough 30m data. Try increasing 'days' or check the coin.")
        else:
            st.write(f"**30m Prediction Score:** {score30}/100")
            st.write(f"**30m Signal:** {signal30}")
            st.write(f"**Swing-Trade Confidence:** {conf30}")
            st.write("Sample 30m Data (first 5 rows):")
            st.write(df30.head())
            plot_chart(df30, coin_id, '30m')

if __name__ == "__main__":
    main()
