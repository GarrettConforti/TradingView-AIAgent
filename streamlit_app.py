import numpy as np
np.NaN = np.nan  # Monkey patch: ensures uppercase NaN exists for pandas_ta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
import ccxt  # For fetching crypto data from exchanges
import requests  # For fetching data from Dexscreener

# -------------------------------
# CORE FUNCTIONS FOR EXCHANGE DATA & TA
# -------------------------------

def get_data(symbol, exchange_name, timeframe='15m', limit=500):
    """
    Fetches OHLCV data for a given trading pair from the specified exchange.
    """
    try:
        # Get the exchange class dynamically (e.g., ccxt.kraken, ccxt.coinbasepro, ccxt.gemini)
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

def compute_indicators(df):
    """
    Computes common technical indicators using pandas_ta.
    """
    df['SMA14'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    return df

def calculate_fibonacci_levels(df, lookback=100):
    """
    Calculates Fibonacci retracement levels using the most recent 'lookback' bars.
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

def analyze_volume_price(df):
    """
    Computes a simple volume-price action signal.
    """
    df['vol_SMA'] = ta.sma(df['volume'], length=14)
    df['price_change'] = df['close'].pct_change()
    df['vol_price_signal'] = np.where(
        (df['volume'] > df['vol_SMA']) & (df['price_change'] > 0), 1,
        np.where((df['volume'] > df['vol_SMA']) & (df['price_change'] < 0), -1, 0)
    )
    return df

def calculate_percentage_wheel(df, fib_levels):
    """
    Aggregates various technical signals into a Prediction Score (0 to 100).
    """
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

    # Bollinger Bands: Price near lower band suggests oversold conditions.
    if latest['close'] < latest['BB_lower']:
        score += 5
    elif latest['close'] > latest['BB_upper']:
        score -= 5

    # Fibonacci retracement: Proximity to key levels can adjust bias.
    tolerance = 0.01 * latest['close']
    if abs(latest['close'] - fib_levels['38.2%']) < tolerance or abs(latest['close'] - fib_levels['50%']) < tolerance:
        score += 5
    if abs(latest['close'] - fib_levels['23.6%']) < tolerance:
        score -= 5

    # Volume/Price Action: Supports the overall trend.
    score += latest['vol_price_signal'] * 5

    return max(0, min(100, score))

def predict_signal(score):
    """
    Converts the Prediction Score into a simple trade signal.
    """
    if score > 55:
        return 'Buy'
    elif score < 45:
        return 'Sell'
    else:
        return 'Neutral'

def explain_signal(df, score, signal, fib_levels):
    """
    Provides a brief explanation for the trade signal.
    """
    latest = df.iloc[-1]
    if signal == "Buy":
        return (f"RSI at {latest['RSI']:.1f} suggests oversold conditions, price is above the 14-period SMA at {latest['SMA14']:.2f}, "
                "and it is near key Fibonacci retracement levels. Volume and price action further support a bullish outlook.")
    elif signal == "Sell":
        return (f"RSI at {latest['RSI']:.1f} indicates overbought conditions, price is below the 14-period SMA at {latest['SMA14']:.2f}, "
                "and it is away from key support levels. Volume and price action imply bearish momentum.")
    else:
        return "Indicators are mixed, resulting in a neutral outlook with no strong bias."

def plot_chart(df, symbol, timeframe):
    """
    Plots the close price with technical indicators.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Close Price')
    ax.plot(df['SMA14'], label='SMA 14', linestyle='--')
    ax.plot(df['BB_upper'], label='BB Upper', linestyle='--')
    ax.plot(df['BB_lower'], label='BB Lower', linestyle='--')
    ax.set_title(f"{symbol} - {timeframe} Chart Analysis")
    ax.legend()
    st.pyplot(fig)

def process_timeframe(symbol, exchange_name, timeframe):
    """
    Processes a given timeframe by fetching data, computing indicators,
    and calculating the Prediction Score, trade signal, and explanation.
    """
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
# DEXSCREENER DATA FUNCTION
# -------------------------------

def get_dexscreener_data(url):
    """
    Converts a Dexscreener URL to its API endpoint and fetches the JSON data.
    Example:
      Input:  https://dexscreener.com/solana/x131b3frgn4b8ue51eyvrnzwutubgom93uryrnteefy
      API:    https://api.dexscreener.com/latest/dex/trading-pairs/solana/x131b3frgn4b8ue51eyvrnzwutubgom93uryrnteefy
    """
    try:
        parts = url.split('/')
        if len(parts) >= 5:
            chain = parts[3]
            pool = parts[4]
            api_url = f"https://api.dexscreener.com/latest/dex/trading-pairs/{chain}/{pool}"
            response = requests.get(api_url)
            if response.status_code == 200:
                return response.json()
            else:
                st.write("Error fetching from Dexscreener API:", response.status_code)
                return None
        else:
            st.write("Invalid Dexscreener URL format.")
            return None
    except Exception as e:
        st.write("Exception in fetching Dexscreener data:", e)
        return None

# -------------------------------
# TOP BUY COINS FUNCTION
# -------------------------------

def get_top_buy_coins_by_quote(exchange_name, quote_currency, timeframe='15m', max_pairs=30):
    """
    Loads all markets from the exchange, filters for pairs ending with the specified quote currency,
    and processes each pair to find those with a 'Buy' signal.
    Returns a DataFrame with the top 10 pairs sorted by Prediction Score.
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        markets = exchange.load_markets()
    except Exception as e:
        st.write("Error loading markets:", e)
        return pd.DataFrame()
    
    # Filter symbols that end with the quote currency (case-insensitive)
    filtered_symbols = [symbol for symbol in markets if symbol.upper().endswith(f"/{quote_currency.upper()}")]
    filtered_symbols = filtered_symbols[:max_pairs]
    
    results = []
    for symbol in filtered_symbols:
        df, score, signal, explanation = process_timeframe(symbol, exchange_name, timeframe)
        if df is not None and signal == "Buy":
            results.append({"Symbol": symbol, "Score": score, "Signal": signal, "Reasoning": explanation})
    
    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    return pd.DataFrame(results[:10])

# -------------------------------
# MAIN APP INTERFACE
# -------------------------------

def main():
    st.title("Crypto Technical Analysis AI Agent")
    st.write("Analyze any trading pair on your chosen exchange to get a Prediction Score, a trade signal, and brief reasoning based on technical analysis.")
    st.write("You can also search for top coins by specifying a quote currency or fetch data directly from Dexscreener.")
    
    # --- Single Pair Analysis Section ---
    st.header("Single Pair Analysis")
    symbol = st.text_input("Trading Pair (e.g., BTC/USD, ETH/USDT)", value="BTC/USD")
    exchange_name = st.selectbox("Select Exchange", options=["kraken", "coinbasepro", "gemini"])
    
    if st.button("Analyze Pair"):
        st.subheader("15-Minute Analysis")
        df15, score15, signal15, explanation15 = process_timeframe(symbol, exchange_name, '15m')
        if df15 is None or df15.empty:
            st.write("No data retrieved for 15-minute timeframe. Check your trading pair and exchange.")
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
            st.write("No data retrieved for 30-minute timeframe. Check your trading pair and exchange.")
        else:
            st.write("Fetched Data (30m) - First 5 rows:")
            st.write(df30.head())
            st.write(f"**30-Minute Prediction Score:** {score30}%")
            st.write(f"**30-Minute Trade Signal:** {signal30}")
            st.write(f"**Reasoning:** {explanation30}")
            plot_chart(df30, symbol, '30m')
    
    # --- Top Buy Coins Search Section ---
    st.header("Top Buy Coins Search")
    quote_currency = st.text_input("Enter Quote Currency (e.g., USD, USDT, EUR)", value="USD")
    if st.button("Find Top Buy Coins"):
        st.write("Analyzing available pairs. Please wait...")
        top_buy_df = get_top_buy_coins_by_quote(exchange_name, quote_currency, timeframe='15m')
        if top_buy_df.empty:
            st.write("No coins currently meet the Buy criteria for the specified quote currency and timeframe.")
        else:
            st.write("Top 10 Buy Recommendations (15m):")
            st.table(top_buy_df)
    
    # --- Dexscreener Data Section ---
    st.header("Dexscreener Data Fetch")
    dex_url = st.text_input("Enter Dexscreener URL (e.g., https://dexscreener.com/solana/x131b3frgn4b8ue51eyvrnzwutubgom93uryrnteefy)", "")
    if st.button("Fetch Dexscreener Data") and dex_url:
        dex_data = get_dexscreener_data(dex_url)
        if dex_data:
            st.write("Dexscreener Data:")
            st.json(dex_data)
        else:
            st.write("Failed to fetch data from Dexscreener.")

if __name__ == "__main__":
    main()
