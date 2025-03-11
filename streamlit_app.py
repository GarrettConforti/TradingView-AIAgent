import numpy as np
np.NaN = np.nan  # Ensure uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests

#############################################
# 1. COINGECKO DATA FUNCTIONS & RESAMPLING
#############################################

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
    For days <= 1, CoinGecko returns ~5-minute intervals.
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
    Resample 5-minute data into OHLCV bars for the specified frequency (e.g. '15T' or '30T').
    Returns a DataFrame with columns: ['open', 'high', 'low', 'close', 'volume'].
    """
    if df.empty:
        return pd.DataFrame()
    ohlc = df["price"].resample(frequency).ohlc()
    vol = df["volume"].resample(frequency).sum()
    df_resampled = ohlc.join(vol)
    return df_resampled

#############################################
# 2. FALLBACK FUNCTIONS: DEXSCREENER & DEXTOOLS
#############################################

def get_dexscreener_data(chain: str, contract: str):
    """
    Attempts to fetch OHLC-like candle data from DexScreener using the chain and contract.
    Expected URL format: https://api.dexscreener.com/latest/dex/trading-pairs/{chain}/{contract}
    Assumes the returned JSON has a key 'chart' with candle data as a list of arrays:
      [timestamp, open, high, low, close, volume]
    """
    url = f"https://api.dexscreener.com/latest/dex/trading-pairs/{chain}/{contract}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "chart" in data and data["chart"]:
            candles = data["chart"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # DexScreener timestamps might be in seconds; adjust as needed:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            return df
    return pd.DataFrame()

def get_dextools_data(chain: str, contract: str):
    """
    Attempts to fetch OHLC-like data from Dextools using the chain and contract.
    (Note: Dextools does not have an official public API; this is an example placeholder.)
    Expected URL format: https://api.dextools.io/app/en/pairs/{chain}/{contract}
    Assumes JSON response with a key 'candles' (this is hypothetical).
    """
    url = f"https://api.dextools.io/app/en/pairs/{chain}/{contract}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "candles" in data and data["candles"]:
            candles = data["candles"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            return df
    return pd.DataFrame()

#############################################
# 3. TECHNICAL INDICATORS & WEIGHTED SCORING
#############################################

def compute_indicators(df: pd.DataFrame):
    """
    Computes advanced indicators:
      - SMA(14), RSI(14), Bollinger Bands (20,2)
      - MACD (12,26,9), Stochastic (14,3,3), ADX (14)
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
    
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = 0
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 'vol_price_signal'] = 1
    df.loc[(df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), 'vol_price_signal'] = -1

    return df

def calculate_fibonacci_levels(df: pd.DataFrame, lookback=50):
    """
    Calculates Fibonacci retracement levels using the last `lookback` bars.
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
    Computes a weighted Prediction Score (0-100) based on:
      - Momentum (RSI, Stoch) ~25 points
      - Trend (SMA, MACD, ADX) ~35 points
      - Volatility (Bollinger) ~10 points
      - Volume/Price ~10 points
      - Fibonacci ~20 points
    """
    latest = df.iloc[-1]
    score = 0
    
    # Momentum: RSI
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10
    # Stochastic
    if latest['STOCH_K'] < 20:
        score += 15
    elif latest['STOCH_K'] > 80:
        score -= 15

    # Trend: SMA
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10
    # Trend: MACD
    if latest['MACD'] > latest['MACD_signal']:
        score += 15
    else:
        score -= 15
    # Trend: ADX
    if latest['ADX'] > 25:
        if latest['MACD_hist'] > 0:
            score += 10
        else:
            score -= 10

    # Volatility: Bollinger Bands
    if latest['close'] < latest['BB_lower']:
        score += 10
    elif latest['close'] > latest['BB_upper']:
        score -= 10

    # Volume/Price Action
    score += latest['vol_price_signal'] * 5

    # Fibonacci Retracement
    close_val = latest['close']
    tolerance = 0.01 * close_val
    if abs(close_val - fib_levels['38.2%']) < tolerance or abs(close_val - fib_levels['50%']) < tolerance:
        score += 10
    if abs(close_val - fib_levels['23.6%']) < tolerance:
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

######################################
# 4. ANALYSIS PIPELINE WITH FALLBACK
######################################

def analyze_coin(coin_id: str, vs_currency='usd', days=1, frequency='15T', lookback=50,
                 fallback_chain: str = None, fallback_contract: str = None):
    """
    Attempts to fetch intraday data from CoinGecko and resample it to the desired frequency.
    If CoinGecko returns no data, and fallback_chain & fallback_contract are provided,
    it will try DexScreener and then Dextools as backups.
    Returns: (DataFrame, score, signal, confidence, fib_levels)
    """
    # Primary: CoinGecko
    df_raw = get_market_chart_data(coin_id, vs_currency, days)
    if not df_raw.empty:
        df_resampled = resample_ohlcv(df_raw, frequency)
    else:
        df_resampled = pd.DataFrame()
    
    # If CoinGecko data is empty and fallback parameters are provided, try DexScreener:
    if df_resampled.empty and fallback_chain and fallback_contract:
        st.write("CoinGecko data not available. Trying DexScreener fallback...")
        df_resampled = get_dexscreener_data(fallback_chain, fallback_contract)
        if df_resampled.empty:
            st.write("DexScreener fallback failed. Trying Dextools fallback...")
            df_resampled = get_dextools_data(fallback_chain, fallback_contract)
    
    if df_resampled.empty or len(df_resampled) < 10:
        return None, None, None, None, None

    df_resampled = compute_indicators(df_resampled)
    fib_levels = calculate_fibonacci_levels(df_resampled, lookback)
    score = calculate_weighted_score(df_resampled, fib_levels)
    signal = predict_trade_signal(score)
    conf = confidence_text(score)
    
    return df_resampled, score, signal, conf, fib_levels

######################################
# 5. CHART PLOTTING FUNCTION
######################################

def plot_chart(df: pd.DataFrame, coin_id: str, timeframe: str):
    """
    Plots the close price with key indicators.
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
# 6. STREAMLIT APP INTERFACE
######################################

def main():
    st.title("Swinger - AI Agent")
    st.write("Analyze a coin's intraday data for swing trading on 15m and 30m timeframes using CoinGecko data. "
             "If CoinGecko data is unavailable, you can provide fallback parameters for DexScreener/Dextools.")
    
    # Primary input for CoinGecko URL
    cg_url = st.text_input("CoinGecko URL (e.g., https://www.coingecko.com/en/coins/immutable-x)", "")
    days = st.number_input("Days of Data (1 to 30)", min_value=1, max_value=30, value=7, step=1)
    frequency_15 = '15T'
    frequency_30 = '30T'
    
    # Fallback parameters (optional)
    st.write("Optional: Provide fallback details if CoinGecko data is unavailable.")
    fallback_chain = st.text_input("Fallback Chain (e.g., solana, ethereum)", "")
    fallback_contract = st.text_input("Fallback Contract Address", "")
    
    if st.button("Run TA Analysis"):
        coin_id = parse_coingecko_coin_id(cg_url)
        if not coin_id:
            st.write("Invalid CoinGecko URL. Must be like https://www.coingecko.com/en/coins/immutable-x")
            return
        
        # 15-Minute Analysis
        st.subheader("15-Minute Chart Analysis")
        df15, score15, signal15, conf15, fib15 = analyze_coin(coin_id, 'usd', days, frequency_15, lookback,
                                                                fallback_chain, fallback_contract)
        if df15 is None or df15.empty:
            st.write("Not enough 15m data. Try adjusting days or check the coin/fallback parameters.")
        else:
            st.write(f"**15m Prediction Score:** {score15}/100")
            st.write(f"**15m Signal:** {signal15}")
            st.write(f"**Swing-Trade Confidence:** {conf15}")
            st.write("Sample 15m Data (first 5 rows):")
            st.write(df15.head())
            plot_chart(df15, coin_id, '15m')
        
        # 30-Minute Analysis
        st.subheader("30-Minute Chart Analysis")
        df30, score30, signal30, conf30, fib30 = analyze_coin(coin_id, 'usd', days, frequency_30, lookback,
                                                                fallback_chain, fallback_contract)
        if df30 is None or df30.empty:
            st.write("Not enough 30m data. Try adjusting days or check the coin/fallback parameters.")
        else:
            st.write(f"**30m Prediction Score:** {score30}/100")
            st.write(f"**30m Signal:** {signal30}")
            st.write(f"**Swing-Trade Confidence:** {conf30}")
            st.write("Sample 30m Data (first 5 rows):")
            st.write(df30.head())
            plot_chart(df30, coin_id, '30m')

if __name__ == "__main__":
    main()
