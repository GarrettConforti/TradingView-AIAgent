import numpy as np
np.NaN = np.nan  # Ensure uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests

######################################
# 1. URL PARSING & DATA FETCH
######################################

def parse_user_url(url: str):
    """
    Determine whether the user URL is a CoinGecko link or a DexScreener link.
    Return a tuple (source, identifier) where:
      source = 'coingecko' or 'dexscreener'
      identifier = coin_id (if coingecko) OR (chain, contract) if dex
    """
    url = url.strip().lower()
    if "coingecko.com" in url and "/coins/" in url:
        # parse coin_id
        parts = url.split('/')
        if 'coins' in parts:
            idx = parts.index('coins')
            if idx + 1 < len(parts):
                coin_id = parts[idx + 1]
                return ('coingecko', coin_id)
        return ('coingecko', None)
    elif "dexscreener.com" in url:
        # parse chain + contract
        # format: https://dexscreener.com/solana/9fmdkqip...
        parts = url.split('/')
        if len(parts) >= 5:
            chain = parts[3]
            contract = parts[4]
            return ('dexscreener', (chain, contract))
        return ('dexscreener', None)
    else:
        return (None, None)

def fetch_coingecko_data(coin_id: str, vs_currency='usd', days=7):
    """
    Fetch intraday data from CoinGecko's /market_chart endpoint.
    Returns a DataFrame with columns ['price', 'volume'] and a DateTime index.
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
        dfp = pd.DataFrame(prices, columns=["timestamp", "price"])
        dfv = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df = pd.merge(dfp, dfv, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        st.write("CoinGecko API Error:", r.status_code)
        return pd.DataFrame()

def fetch_dexscreener_data(chain: str, contract: str):
    """
    Attempt to fetch candle data from DexScreener's pool API:
      https://api.dexscreener.com/latest/dex/trading-pairs/{chain}/{contract}
    We expect a 'chart' key with candle arrays: [timestamp_s, open, high, low, close, volume].
    """
    url = f"https://api.dexscreener.com/latest/dex/trading-pairs/{chain}/{contract}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "chart" in data and isinstance(data["chart"], list) and data["chart"]:
            candles = data["chart"]  # each item: [ts_s, open, high, low, close, volume]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # DexScreener timestamps are in seconds
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            return df
        else:
            st.write("DexScreener: no 'chart' data found.")
            return pd.DataFrame()
    else:
        st.write("DexScreener API Error:", r.status_code)
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, frequency: str):
    """
    Resample DataFrame with columns: open, high, low, close, volume to the desired frequency (e.g. '15T').
    If the DF was from CoinGecko, we only have 'price' & 'volume', so we do a different approach.
    """
    if df.empty:
        return pd.DataFrame()

    required_cols = set(["open", "high", "low", "close", "volume"])
    if required_cols.issubset(df.columns):
        # We already have OHLCV
        ohlc = df[["open", "high", "low", "close"]].resample(frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        vol = df["volume"].resample(frequency).sum()
        df_resampled = pd.concat([ohlc, vol], axis=1)
        return df_resampled
    else:
        # This must be CoinGecko data with columns ["price", "volume"]
        ohlc = df["price"].resample(frequency).ohlc()
        vol = df["volume"].resample(frequency).sum()
        df_resampled = ohlc.join(vol)
        return df_resampled

#############################################
# 2. ADVANCED INDICATORS & WEIGHTED SCORING
#############################################

def compute_indicators(df: pd.DataFrame):
    """
    Compute advanced indicators (SMA, RSI, Bollinger, MACD, Stoch, ADX, Volume/Price).
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

    # Trend: SMA14
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

    # Volatility: Bollinger
    if latest['close'] < latest['BB_lower']:
        score += 10
    elif latest['close'] > latest['BB_upper']:
        score -= 10

    # Volume/Price
    score += latest['vol_price_signal'] * 5

    # Fibonacci
    close_val = latest['close']
    tolerance = 0.01 * close_val
    if abs(close_val - fib_levels['38.2%']) < tolerance or abs(close_val - fib_levels['50%']) < tolerance:
        score += 10
    if abs(close_val - fib_levels['23.6%']) < tolerance:
        score -= 10

    return max(0, min(100, score))

def predict_trade_signal(score: float):
    if score >= 70:
        return "Buy"
    elif score <= 30:
        return "Sell"
    else:
        return "Neutral"

def confidence_text(score: float):
    if score >= 80:
        return "High confidence in a bullish swing trade."
    elif score >= 60:
        return "Moderate confidence in a bullish swing trade."
    elif score >= 40:
        return "Mixed signals; caution advised."
    else:
        return "Low confidence; market appears bearish."

######################################
# 3. UNIFIED ANALYSIS FUNCTION
######################################

def analyze_url(url: str, days=7, frequency='15T', lookback=50):
    """
    Parse the user URL. If it's a CoinGecko link, fetch from CoinGecko.
    If it's a DexScreener link, fetch from DexScreener.
    Then resample to 'frequency', compute indicators, and produce a weighted score.
    """
    source, identifier = parse_user_url(url)
    if source == 'coingecko':
        # identifier = coin_id
        if not identifier:
            return None, None, None, None, None
        df_raw = fetch_coingecko_data(identifier, 'usd', days)
        if df_raw.empty:
            return None, None, None, None, None
        df_resampled = resample_ohlcv(df_raw, frequency)
    elif source == 'dexscreener':
        # identifier = (chain, contract)
        if not identifier or not isinstance(identifier, tuple):
            return None, None, None, None, None
        chain, contract = identifier
        df_raw = fetch_dexscreener_data(chain, contract)
        if df_raw.empty:
            return None, None, None, None, None
        # DexScreener might already have open/high/low/close/volume
        df_resampled = resample_ohlcv(df_raw, frequency)
    else:
        # unrecognized URL
        return None, None, None, None, None

    if df_resampled.empty or len(df_resampled) < 10:
        return None, None, None, None, None

    # rename columns if needed to standardize
    # we expect columns: open, high, low, close, volume
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df_resampled.columns:
            return None, None, None, None, None

    # compute advanced indicators
    df_resampled = compute_indicators(df_resampled)
    fib_levels = calculate_fibonacci_levels(df_resampled, lookback)
    score = calculate_weighted_score(df_resampled, fib_levels)
    signal = predict_trade_signal(score)
    conf = confidence_text(score)
    return df_resampled, score, signal, conf, fib_levels

def plot_chart(df: pd.DataFrame, title_str: str):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(title_str)
    ax.legend()
    st.pyplot(fig)

######################################
# 4. STREAMLIT APP
######################################

def main():
    st.title("Swinger - AI Agent")
    st.write("Enter either a CoinGecko URL (e.g. https://www.coingecko.com/en/coins/immutable-x) or a DexScreener URL (e.g. https://dexscreener.com/solana/<contract>).")
    st.write("We'll parse and fetch the data, then do 15m & 30m TA with advanced indicators and a weighted score.")
    
    user_url = st.text_input("CoinGecko or DexScreener URL", "")
    days = st.number_input("Days of Data (CoinGecko only)", min_value=1, max_value=30, value=7, step=1,
                           help="For DexScreener, 'days' is ignored. For CoinGecko, days up to 30.")
    
    if st.button("Run TA Analysis"):
        # 15-Minute
        st.subheader("15-Minute Chart Analysis")
        df15, score15, signal15, conf15, fib15 = analyze_url(user_url, days, '15T', 50)
        if df15 is None or df15.empty:
            st.write("No 15m data. Possibly the URL is invalid or data not available.")
        else:
            st.write(f"**15m Score:** {score15}/100")
            st.write(f"**15m Signal:** {signal15}")
            st.write(f"**Confidence:** {conf15}")
            st.write("Sample 15m data (first 5 rows):")
            st.write(df15.head())
            plot_chart(df15, "15m Chart Analysis")
        
        # 30-Minute
        st.subheader("30-Minute Chart Analysis")
        df30, score30, signal30, conf30, fib30 = analyze_url(user_url, days, '30T', 50)
        if df30 is None or df30.empty:
            st.write("No 30m data. Possibly the URL is invalid or data not available.")
        else:
            st.write(f"**30m Score:** {score30}/100")
            st.write(f"**30m Signal:** {signal30}")
            st.write(f"**Confidence:** {conf30}")
            st.write("Sample 30m data (first 5 rows):")
            st.write(df30.head())
            plot_chart(df30, "30m Chart Analysis")

if __name__ == "__main__":
    main()
