import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # <-- Using pandas_ta for all indicators

# 1) Fetch data from TradingView if possible, else yfinance
def get_data(ticker, exchange):
    """
    Tries to fetch 15-minute historical data via tvDatafeed.
    If that fails, falls back to yfinance (though yfinance may have limited intraday data).
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()  # If needed, pass your TradingView username/password
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=Interval.in_15_minute, n_bars=500)
        st.write("Using TradingView data.")
    except Exception as e:
        st.write("Failed to fetch from TradingView, using yfinance as fallback.")
        import yfinance as yf
        # yfinance might not support 15-minute data for all symbols; adjust if needed
        df = yf.download(ticker, interval='15m', period='5d')
    
    # Standardize column names for consistency
    df.rename(columns={
        'Open': 'open', 
        'High': 'high', 
        'Low': 'low', 
        'Close': 'close', 
        'Volume': 'volume'
    }, inplace=True, errors='ignore')
    
    return df

# 2) Compute Indicators with pandas_ta
def compute_indicators(df):
    """
    Applies RSI, SMA(14), and Bollinger Bands (period=20, std=2) using pandas_ta.
    """
    # --- SMA (14) ---
    df['SMA14'] = ta.sma(df['close'], length=14)
    
    # --- RSI (14) ---
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # --- Bollinger Bands (period=20, std=2) ---
    # pandas_ta.bbands() returns a DataFrame with columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0
    bb = ta.bbands(df['close'], length=20, std=2)
    # Rename them for convenience
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    
    return df

# 3) Calculate Fibonacci Retracement Levels
def calculate_fibonacci_levels(df, lookback=100):
    """
    Finds the highest high and lowest low over the last `lookback` bars,
    then calculates common Fibonacci retracement levels.
    """
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

# 4) Analyze Volume & Price Action
def analyze_volume_price(df):
    """
    Creates a volume-based signal:
      +1 if volume > vol_SMA and price up,
      -1 if volume > vol_SMA and price down,
       0 otherwise.
    """
    # Compute a 14-bar SMA of volume
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    
    # Calculate % price change from the previous bar
    df['price_change'] = df['close'].pct_change()
    
    # Volume/Price signal
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where(
            (df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0
        )
    )
    return df

# 5) Aggregate All Signals into a Percentage Score
def calculate_percentage_wheel(df, fib_levels):
    """
    Aggregates signals from RSI, SMA, Bollinger Bands, Fibonacci levels, and volume/price action
    into a score from 0 to 100 (50 = neutral).
    """
    latest = df.iloc[-1]
    score = 50  # Start neutral
    
    # --- RSI contribution ---
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10
    
    # --- SMA (14) contribution ---
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10
    
    # --- Bollinger Bands contribution ---
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5
    
    # --- Fibonacci Retracement contribution ---
    close = latest['close']
    tolerance = 0.01 * close  # 1% tolerance
    # If near support levels (38.2% or 50%), add a small bullish bias
    if abs(close - fib_levels['38.2%']) < tolerance or abs(close - fib_levels['50%']) < tolerance:
        score += 5
    # If near resistance level (23.6%), subtract
    if abs(close - fib_levels['23.6%']) < tolerance:
        score -= 5
    
    # --- Volume/Price Action contribution ---
    score += latest['vol_price_signal'] * 5
    
    # Bound the score between 0 and 100
    score = max(0, min(100, score))
    return score

# 6) Convert Score to a Simple Buy/Sell/Neutral Signal
def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

# 7) Plot a Simple Chart
def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{ticker} - 15 Minute Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# 8) Streamlit UI
def main():
    st.title("AI Agent for Technical Analysis (pandas_ta Version)")
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
        
        # Plot the chart
        plot_chart(df, ticker)

if __name__ == "__main__":
    main()
