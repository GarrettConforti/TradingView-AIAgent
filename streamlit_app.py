import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
import ccxt  # For fetching crypto data

# Function to fetch crypto data using ccxt from the selected exchange and timeframe
def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    try:
        # Dynamically get the exchange class (e.g., ccxt.binance or ccxt.kraken)
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.write(f"Data fetched from {exchange_name} for {timeframe} timeframe for {symbol}.")
    except Exception as e:
        st.write(f"Error fetching data for {symbol} in {timeframe} timeframe:", e)
        df = pd.DataFrame()
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

# Aggregate technical signals into a Prediction Score (0 to 100)
def calculate_percentage_wheel(df, fib_levels):
    latest = df.iloc[-1]
    score = 50  # Neutral starting point
    
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

    # Fibonacci retracement contribution (using a tolerance of 1% of close)
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price Action contribution
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

# Convert the prediction score into a simple trade signal
def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

# Plot the chart with technical indicators
def plot_chart(df, symbol, timeframe):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{symbol} - {timeframe} Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# Process a given timeframe: fetch data, compute indicators, calculate score and signal
def process_timeframe(symbol, exchange_name, timeframe):
    df = get_data(symbol, exchange_name, timeframe, limit=500)
    if df.empty:
        return None, None, None
    df = compute_indicators(df)
    fib_levels = calculate_fibonacci_levels(df, lookback=100)
    df = analyze_volume_price(df)
    score = calculate_percentage_wheel(df, fib_levels)
    signal = predict_signal(score)
    return df, score, signal

# New function: Get Top 10 Coins with a Buy Rating based on 15-minute analysis
def get_top_buy_coins(exchange_name, timeframe='15m'):
    # Predefined list of popular coins (adjust according to exchange)
    if exchange_name == "binance":
        coin_list = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "LTC/USDT", "DOT/USDT", "AVAX/USDT"]
    elif exchange_name == "kraken":
        coin_list = ["XBT/USD", "ETH/USD", "ADA/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LTC/USD", "DOT/USD", "BCH/USD", "LINK/USD"]
    else:
        coin_list = []
    
    results = []
    for coin in coin_list:
        df, score, signal = process_timeframe(coin, exchange_name, timeframe)
        if df is not None and not df.empty:
            if signal == "Buy":
                results.append({"Coin": coin, "Score": score, "Signal": signal})
    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    return pd.DataFrame(results[:10])

# Main Streamlit app interface
def main():
    st.title("Crypto Technical Analysis AI Agent")
    st.write("Enter a crypto trading pair and select an exchange to analyze prediction scores for different timeframes.")
    st.write("For example, for Binance use **BTC/USDT**; for Kraken use **XBT/USD**.")
    
    symbol = st.text_input("Trading Pair", value="BTC/USDT")
    exchange_name = st.selectbox("Select Exchange", options=["binance", "kraken"])
    
    if st.button("Analyze"):
        st.subheader("15-Minute Analysis")
        df15, score15, signal15 = process_timeframe(symbol, exchange_name, '15m')
        if df15 is None or df15.empty:
            st.write("No data retrieved for 15-minute timeframe. Please check your trading pair symbol and exchange selection.")
        else:
            st.write("Fetched Data (15m) - First 5 rows:")
            st.write(df15.head())
            st.write(f"**15-Minute Prediction Score:** {score15}%")
            st.write(f"**15-Minute Trade Signal:** {signal15}")
            plot_chart(df15, symbol, '15m')
        
        st.subheader("30-Minute Analysis")
        df30, score30, signal30 = process_timeframe(symbol, exchange_name, '30m')
        if df30 is None or df30.empty:
            st.write("No data retrieved for 30-minute timeframe. Please check your trading pair symbol and exchange selection.")
        else:
            st.write("Fetched Data (30m) - First 5 rows:")
            st.write(df30.head())
            st.write(f"**30-Minute Prediction Score:** {score30}%")
            st.write(f"**30-Minute Trade Signal:** {signal30}")
            plot_chart(df30, symbol, '30m')
        
        st.subheader("Top 10 Buy Recommendations (15-Minute Analysis)")
        top_buy_df = get_top_buy_coins(exchange_name, timeframe='15m')
        if top_buy_df.empty:
            st.write("No coins currently meet the Buy criteria based on the 15-minute analysis.")
        else:
            st.write("Top 10 Buy Recommendations:")
            st.table(top_buy_df)

if __name__ == "__main__":
    main()
