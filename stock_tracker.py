import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Initialize session state for stop tracking button
if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = False

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo", interval="1h")  # Fetch last 3 months data with hourly intervals
    return df

# Function to calculate technical indicators
def add_technical_indicators(df):
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["BB_High"], df["BB_Mid"], df["BB_Low"] = ta.volatility.bollinger_hband(df["Close"]), ta.volatility.bollinger_mavg(df["Close"]), ta.volatility.bollinger_lband(df["Close"])
    return df

# Function to get market sentiment from financial news
def get_market_sentiment(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = soup.find_all("h3")
        sentiment_score = 0
        for h in headlines[:5]:  # Analyze the top 5 headlines
            text = h.text.lower()
            if any(word in text for word in ["rises", "soars", "strong", "bullish", "positive"]):
                sentiment_score += 1
            elif any(word in text for word in ["drops", "falls", "weak", "bearish", "negative"]):
                sentiment_score -= 1
        return sentiment_score
    else:
        return 0  # Neutral sentiment if unable to fetch

# Function to generate buy/sell signals
def generate_signals(df):
    # Initialize signals with NaN values
    df["Buy_Signal"] = np.nan
    df["Sell_Signal"] = np.nan

    for i in range(1, len(df)):  # Avoid index errors at i=0
        buy_signal = None
        sell_signal = None

        # Buy condition: RSI below 30 & MACD above MACD_Signal
        if df["RSI"].iloc[i] < 30 and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]:
            buy_signal = df["Close"].iloc[i]

        # Sell condition: RSI above 70 & MACD below MACD_Signal
        elif df["RSI"].iloc[i] > 70 and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]:
            sell_signal = df["Close"].iloc[i]

        # Store signals in DataFrame
        df.at[df.index[i], "Buy_Signal"] = buy_signal
        df.at[df.index[i], "Sell_Signal"] = sell_signal

    return df

# Streamlit UI
st.title("📈 Real-time Stock Tracker with AI-based Signals")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Button to stop tracking
if st.button("Stop Tracking"):
    st.session_state.stop_tracking = True

st.write("Tracking will refresh every 15 seconds...")

placeholder = st.empty()  # Placeholder for updating content dynamically

# Real-time tracking loop
while not st.session_state.stop_tracking:
    df = get_stock_data(ticker)
    df = add_technical_indicators(df)
    df = generate_signals(df)  # Add buy/sell signals
    sentiment = get_market_sentiment(ticker)
    
    # Create plot
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlesticks"
    ))

    # EMA 20
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color='blue')))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], mode="lines", name="BB High", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], mode="lines", name="BB Low", line=dict(color='red')))

    # Buy Signals (🔵)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Buy_Signal"], mode="markers", name="BUY Signal",
        marker=dict(symbol="triangle-up", size=10, color="blue")
    ))

    # Sell Signals (🔴)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Sell_Signal"], mode="markers", name="SELL Signal",
        marker=dict(symbol="triangle-down", size=10, color="red")
    ))

    # Update layout
    fig.update_layout(title=f"{ticker} Stock Performance", xaxis_rangeslider_visible=False)

    # Display chart and signals
    with placeholder.container():
        st.plotly_chart(fig, key=f"chart_{time.time()}")  # Ensure a unique key to prevent duplicate IDs
        st.write(f"**Market Sentiment Score:** {sentiment} (Higher is better)")
    
    time.sleep(15)  # Refresh every 15 seconds
    st.rerun()  # Forces a rerun for real-time updates
