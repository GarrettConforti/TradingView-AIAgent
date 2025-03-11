import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
import ccxt  # For fetching crypto data

# -------------------------------
# CORE FUNCTIONS
# -------------------------------

# Fetch data using ccxt from the selected exchange for the given timeframe
def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    try:
        # Get the exchange class dynamically (e.g., ccxt.kraken, ccxt.coinbasepro, ccxt.gemini)
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.write(f"Data fetched from {exchange_name} for {timeframe} timeframe for {symbol}.")
    except Exception as e:
        st.write(f"Error fetching data for {symbol} in {timeframe} timeframe:", e)
        df = pd.DataFrame()
    return df

# Calculate technical indicators using pandas_ta
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

# Analyze volume and price action to create a simple signal component
def analyze_volume_price(df):
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

# Aggregate the various signals into a Prediction Score (0-100)
def calculate_percentage_wheel(df, fib_levels):
    latest = df.iloc[-1]
    score = 50  # Neutral starting point

    # RSI: Oversold adds bullish bias; overbought adds bearish bias.
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10

    # SMA: Price above SMA indicates bullish trend.
    if latest['close'] > latest['SMA14']:
        score += 10
    else:
        score -= 10

    # Bollinger Bands: If price is near the lower band, it suggests oversold conditions.
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    # Fibonacci: Being near key retracement levels can add/subtract a small bias.
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price action: A positive volume-price signal supports bullishness.
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

# Convert the Prediction Score into a simple trade signal.
def predict_signal(score):
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

# Generate a short explanation based on key indicators
def explain_signal(df, score, signal, fib_levels):
    latest = df.iloc[-1]
    explanation = ""
    if signal == "Buy":
        explanation = (
            f"The RSI is at {latest['RSI']:.1f}, suggesting oversold conditions, "
            f"and the price is above the 14-period SMA ({latest['SMA14']:.2f}), "
            f"with the price near key Fibonacci levels. Volume action supports upward momentum."
        )
    elif signal == "Sell":
        explanation = (
            f"The RSI is at {latest['RSI']:.1f}, suggesting overbought conditions, "
            f"and the price is below the 14-period SMA ({latest['SMA14']:.2f}), "
            f"with the price away from key support levels. Volume action indicates bearish pressure."
        )
    else:
        explanation = (
            "The indicators are mixed, with no strong directional bias. "
            "RSI, SMA, and Bollinger Bands suggest a neutral trend."
        )
    return explanation

# Plot chart with technical indicators
def plot_chart(df, symbol, timeframe):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{symbol} - {timeframe} Chart Analysis")
    ax.legend()
    st.pyplot(fig)

# Process a given timeframe: fetch data, compute indicators, calculate score, signal, and explanation
def process_timeframe(symbol, exchange_name, timeframe):
    df = get_data(symbol, exchange_name, timeframe, limit=500)
    if df.empty:
        return None, None, None, None
    df = compute_indicators(df)
    fib_levels = calculate_fibonacci_levels(df, lookback=100)
    df = analyze_volume_price(df)
    score = calculate_percentage_wheel(df, fib_levels)
    signal = predict_signal(score)
    explanation = explain_signal(df, score, signal, fib_levels)
    return df, score, signal, explanation

# -------------------------------
# TOP BUY COINS FUNCTION (CASTING WIDE NET)
# -------------------------------

def get_top_buy_coins_by_quote(exchange_name, quote_currency, timeframe='15m', max_pairs=30):
    """
    Load all markets from the exchange, filter for pairs ending with the given quote_currency,
    then process each pair. Returns a DataFrame with the top 10 pairs that have a 'Buy' signal.
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        markets = exchange.load_markets()
    except Exception as e:
        st.write("Error loading markets:", e)
        return pd.DataFrame()
    
    # Filter symbols that end with f"/{quote_currency}" (case-insensitive)
    filtered_symbols = [symbol for symbol in markets if symbol.upper().endswith(f"/{quote_currency.upper()}")]
    
    # Limit the number of pairs to analyze (to avoid excessive API calls)
    filtered_symbols = filtered_symbols[:max_pairs]
    
    results = []
    for symbol in filtered_symbols:
        df, score, signal, explanation = process_timeframe(symbol, exchange_name, timeframe)
        if df is not None and signal == "Buy":
            results.append({"Symbol": symbol, "Score": score, "Signal": signal, "Reasoning": explanation})
    
    # Sort results by score descending and return top 10
    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    return pd.DataFrame(results[:10])

# -------------------------------
# MAIN APP INTERFACE
# -------------------------------

def main():
    st.title("Crypto Technical Analysis AI Agent")
    st.write("Analyze any trading pair on your chosen exchange and get a Prediction Score and a Buy/Sell signal with brief reasoning.")
    st.write("You can also search for top coins by specifying a quote currency (e.g., USD, USDT, EUR) to cast the widest net for up-to-date data.")
    
    # Section for single pair analysis
    st.header("Single Pair Analysis")
    symbol = st.text_input("Trading Pair (e.g., BTC/USD, ETH/USDT)", value="BTC/USD")
    exchange_name = st.selectbox("Select Exchange", options=["kraken", "coinbasepro", "gemini"])
    
    if st.button("Analyze Pair"):
        st.subheader("15-Minute Analysis")
        df15, score15, signal15, explanation15 = process_timeframe(symbol, exchange_name, '15m')
        if df15 is None or df15.empty:
            st.write("No data retrieved for 15-minute timeframe. Please check your trading pair symbol and exchange selection.")
        else:
            st.write("Fetched Data (15m) - First 5 rows:")
            st.write(df15.head())
            st.write(f"**15-Minute Prediction Score:** {score15}%")
            st.write(f"**15-Minute Trade Signal:** {signal15}")
            st.write(f"**Reasoning:** {explanation15}")
            plot_chart(df15, symbol, '15m')
        
        st.subheader("30-Minute Analysis")
        df30, score30, signal30, explanation30 = process_timeframe(symbol, exchange_name, '30m')
        if df30 is None or df30.empty:
            st.write("No data retrieved for 30-minute timeframe. Please check your trading pair symbol and exchange selection.")
        else:
            st.write("Fetched Data (30m) - First 5 rows:")
            st.write(df30.head())
            st.write(f"**30-Minute Prediction Score:** {score30}%")
            st.write(f"**30-Minute Trade Signal:** {signal30}")
            st.write(f"**Reasoning:** {explanation30}")
            plot_chart(df30, symbol, '30m')
    
    # Section for top buy coins search
    st.header("Top Buy Coins Search")
    quote_currency = st.text_input("Enter Quote Currency (e.g., USD, USDT, EUR)", value="USD")
    if st.button("Find Top Buy Coins"):
        st.write("Please wait while we analyze available pairs...")
        top_buy_df = get_top_buy_coins_by_quote(exchange_name, quote_currency, timeframe='15m')
        if top_buy_df.empty:
            st.write("No coins currently meet the Buy criteria for the specified quote currency and timeframe.")
        else:
            st.write("Top 10 Buy Recommendations (15m):")
            st.table(top_buy_df)

if __name__ == "__main__":
    main()
