import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Global flag for stopping the tracking
stop_tracking = False

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
    latest = df.iloc[-1]
    signal = "HOLD"
    if latest["RSI"] < 30 and latest["MACD"] > latest["MACD_Signal"]:
        signal = "🔵 STRONG BUY"
    elif latest["RSI"] > 70 and latest["MACD"] < latest["MACD_Signal"]:
        signal = "🔴 STRONG SELL"
    return signal

# Streamlit UI
st.title("📈 Real-time Stock Tracker with AI-based Signals")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Button to stop tracking
if st.button("Stop Tracking"):
    stop_tracking = True

st.write("Tracking will refresh every 15 seconds...")

placeholder = st.empty()

while not stop_tracking:
    df = get_stock_data(ticker)
    df = add_technical_indicators(df)
    signal = generate_signals(df)
    sentiment = get_market_sentiment(ticker)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlesticks"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], mode="lines", name="BB High", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], mode="lines", name="BB Low", line=dict(color='red')))
    
    # Update layout
    fig.update_layout(title=f"{ticker} Stock Performance", xaxis_rangeslider_visible=False)
    
    # Display chart and signals
    with placeholder.container():
        st.plotly_chart(fig)
        st.write(f"**Technical Indicator Signal:** {signal}")
        st.write(f"**Market Sentiment Score:** {sentiment} (Higher is better)")
    
    time.sleep(15)  # Refresh every 15 seconds
