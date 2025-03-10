import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Technical analysis libraries
try:
    import talib
except ImportError:
    import pandas_ta as ta

# For data retrieval
def get_data(ticker, exchange):
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()  # Optionally pass username/password here
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=Interval.in_15_minute, n_bars=500)
        st.write("Using TradingView data.")
    except Exception as e:
        st.write("Failed to fetch from TradingView, using yfinance as fallback.")
        import yfinance as yf
        df = yf.download(ticker, interval='15m', period='5d')
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    return df

def compute_indicators(df):
    try:
        df['SMA14'] = talib.SMA(df['close'], timeperiod=14)
    except Exception:
        df['SMA14'] = df['close'].rolling(window=14).mean()
    try:
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    except Exception:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
    try:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
    except Exception:
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * std
        df['BB_lower'] = df['BB_middle'] - 2 * std
    return df

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

def analyze_volume_price(df):
    try:
        df['vol_SMA'] = talib.SMA(df['volume'], timeperiod=14)
    except Exception:
        df['vol_SMA'] = df['volume'].rolling(window=14).mean()
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

def calculate_percentage_wheel(df, fib_levels):
    latest = df.iloc[-1]
    score = 50  # Start neutral

    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    tolerance = 0.01 * latest['close']
    close = latest['close']
    if abs(close - fib_levels['38.2%']) < tolerance or abs(close - fib_levels['50%']) < tolerance:
        score += 5
    if abs(close - fib_levels['23.6%']) < tolerance:
        score -= 5

    score += latest['vol_price_signal'] * 5
    score = max(0, min(100, score))
    return score

def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{ticker} - 15 Minute Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# --- Streamlit Interface ---
st.title("AI Agent for Technical Analysis")
st.write("Enter a ticker symbol and exchange to see the prediction based on TradingView indicators.")

ticker = st.text_input("Ticker Symbol (e.g., BTCUSDT, AAPL)", value="BTCUSDT")
exchange = st.text_input("Exchange (e.g., BINANCE, NASDAQ)", value="BINANCE")

if st.button("Analyze"):
    df = get_data(ticker, exchange)
    if df is not None and not df.empty:
        df = compute_indicators(df)
        fib_levels = calculate_fibonacci_levels(df, lookback=100)
        df = analyze_volume_price(df)
        score = calculate_percentage_wheel(df, fib_levels)
        signal = predict_signal(score)
        st.write(f"Prediction score: {score}%")
        st.write(f"Trade Signal: **{signal}**")
        plot_chart(df, ticker)
    else:
        st.write("No data retrieved. Please check the ticker and exchange.")
