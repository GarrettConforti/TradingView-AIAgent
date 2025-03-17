import ccxt
import pandas as pd
import numpy as np
import ta  # Technical analysis library
import matplotlib.pyplot as plt

# --- Step 1: Data Retrieval using ccxt ---
def get_ohlcv(exchange_id, symbol, timeframe, limit=500):
    """
    Retrieves OHLCV data using ccxt.
    exchange_id: string identifier for the exchange (e.g., 'kraken', 'coinbasepro', 'gemini')
    symbol: trading pair symbol as recognized by the exchange (e.g., 'BTC/USD')
    timeframe: timeframe string (e.g., '15m', '30m')
    limit: number of data points to retrieve
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
    })
    # Many exchanges require you to use uppercase symbols and sometimes different formats.
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# --- Step 2: Calculate Technical Indicators ---
def add_indicators(df):
    # RSI Calculation (using a 14-period window)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD Calculation
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands Calculation
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    
    # Fibonacci retracement levels computed using the highest high and lowest low
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Fib_23.6'] = high - 0.236 * diff
    df['Fib_38.2'] = high - 0.382 * diff
    df['Fib_50.0'] = high - 0.500 * diff
    df['Fib_61.8'] = high - 0.618 * diff
    
    # Price Action: Example simple pin bar detection
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['PinBar'] = np.where((df['Body'] < 0.3 * df['Range']) & ((df['High'] - np.maximum(df['Close'], df['Open'])) > 0.6 * df['Range']),
                              1, 0)
    
    return df

# --- Step 3: Signal Generation ---
def generate_signals(df):
    df['Signal'] = 0

    # Example buy signal: oversold RSI, bullish MACD, price near Fibonacci support, and a pin bar
    buy_condition = (
        (df['RSI'] < 30) &
        (df['MACD'] > df['MACD_Signal']) &
        (df['Close'] <= df['Fib_38.2']) &
        (df['PinBar'] == 1)
    )
    
    # Example sell signal: overbought RSI, bearish MACD, or price hitting upper Bollinger Band
    sell_condition = (
        (df['RSI'] > 70) &
        (df['MACD'] < df['MACD_Signal']) |
        (df['Close'] >= df['BB_High'])
    )
    
    df.loc[buy_condition, 'Signal'] = 1   # Buy
    df.loc[sell_condition, 'Signal'] = -1  # Sell
    return df

# --- Step 
