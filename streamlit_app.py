import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
import ccxt  # For fetching crypto data

# Function to fetch crypto data using ccxt from the selected exchange
def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    try:
        # Get the exchange class dynamically (e.g., ccxt.binance or ccxt.kraken)
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        # Fetch OHLCV data: each row is [timestamp, open, high, low, close, volume]
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
        '0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100%': low
    }
    return levels

# Analyze volume and price action to generate a signal component
def analyze_volume_price(df):
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

# Aggregate indicator signals into a percentage score (0 to 100)
def calculate_percentage_wheel(df, fib_levels):
    latest = df.iloc[-1]
    score = 50  # Start neutral at 50%
    
    # RSI contribution
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    # SMA contribution
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    # Bollinger Bands contribution
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    # Fibonacci retracement contribution (with a tolerance of 1% of the close)
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price Action contribution
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

# Convert the percentage score into a simple trade signal
def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

# Plot the chart with technical indicators
def plot_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{symbol} - 15 Minute Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# Main Streamlit interface
def main():
    st.title("Crypto Technical Analysis AI Agent")
    st.write("Enter a crypto trading pair and select an exchange to analyze 15-minute data.")
    st.write("For example, for Binance use **BTC/USDT**; for Kraken use **XBT/USD**.")
    
    symbol = st.text_input("Trading Pair", value="BTC/USDT")
    exchange_name = st.selectbox("Select Exchange", options=["binance", "kraken"])
    
    if st.button("Analyze"):
        df = get_data(symbol, exchange_name, timeframe='15m', limit=500)
        if df.empty:
            st.write("No data retrieved. Please check your trading pair symbol and exchange selection.")
            return
        
        st.write("Fetched Data (first 5 rows):")
        st.write(df.head())
        
        df = compute_indicators(df)
        fib_levels = calculate_fibonacci_levels(df, lookback=100)
        df = analyze_volume_price(df)
        
        score = calculate_percentage_wheel(df, fib_levels)
        signal = predict_signal(score)
        
        st.write(f"**Prediction Score:** {score}%")
        st.write(f"**Trade Signal:** {signal}")
        
        plot_chart(df, symbol)

if __name__ == "__main__":
    main()
