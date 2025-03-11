import numpy as np
np.NaN = np.nan  # Monkey patch: create uppercase NaN for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
import ccxt  # To fetch crypto data

# Function to fetch crypto data using ccxt from the selected exchange
def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    try:
        # Dynamically get the exchange class from ccxt (e.g., ccxt.binance or ccxt.kraken)
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.write(f"Data fetched from {exchange_name}.")
    except Exception as e:
        st.write("Error fetching data:", e)
        df = pd.DataFrame()  # Return empty DataFrame on error
    return df

# Compute technical indicators using pandas_ta
def compute_indicators(df):
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    return df

# Calculate Fibonacci retracement levels using the most recent 100 bars
def calculate_fibonacci_levels(df, lookback=100):
    recent = df[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    levels = {
        
