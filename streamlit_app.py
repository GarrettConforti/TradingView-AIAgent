import numpy as np
np.NaN = np.nan  # Monkey patch: create uppercase NaN for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # Using pandas_ta for technical indicators

# 1) Fetch data (try TradingView via tvDatafeed, fallback to yfinance)
def get_data(ticker, exchange):
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()  # Optionally pass your TradingView credentials if needed
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=Interval.in_15_minute, n_bars=500)
        st.write("Using TradingView data.")
    except Exception as e:
        st.write("Failed to fetch from TradingView, using yfinance as fallback.")
        import yfinance as yf
        df = yf.download(ticker, interval='15m', period='5d')
    
    # Standardize column names
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True, errors='ignore')
    return df

# 2) Compute technical indicators using pandas_ta
def compute_indicators(df):
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    return df

# 3) Calculate Fibonacci retracement levels based on recent data
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

# 4) Analyze volume and price action
def analyze_volume_price(df):
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

# 5) Aggregate signals into a percentage score (0 to 100)
def calculate_percentage_wheel(df, fib_levels):
    latest = df.iloc[-1]
    score = 50  # Start neutral
    
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
    
    # Fibonacci Retracement contribution (using a 1% tolerance)
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5
    
    # Volume/Price Action contribution
    score += latest['vol_price_signal'] * 5
    
    # Ensure score is between 0 and 100
    score = max(0, min(100, score))
    return score

# 6) Convert percentage score to a simple trade signal
def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

# 7) Plot the chart with indicators
def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{ticker} - 15 Minute Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# 8) Streamlit User Interface
def main():
    st.title("AI Agent for Technical Analysis")
    st.write("Enter a ticker symbol and exchange to see the prediction based on multiple indicators.")
    
    ticker = st.text_input("Ticker Symbol (e.g., BTCUSDT, AAPL)", value="BTCUSDT")
    exchange = st.text_input("Exchange (e.g., BINANCE, NASDAQ)", value="BINANCE")
    
    if st.button("Analyze"):
        df = get_data(ticker, exchange)
        if df is None or df.empty:
            st.write("No data retrieved. Please check the ticker and exchange.")
            return
        
        df = compute_indicators(df)
        fib_levels = calculate_fibonacci_levels(df, lookback=100)
        df = analyze_volume_price(df)
        
        score = calculate_percentage_wheel(df, fib_levels)
        signal = predict_signal(score)
        
        st.write(f"**Prediction Score:** {score}%")
        st.write(f"**Trade Signal:** {signal}")
        
        plot_chart(df, ticker)

if __name__ == "__main__":
    main()
