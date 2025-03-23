import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np  # Import numpy, although it's not directly used, it's often needed with pandas/ta
import ta
import time
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Initialize session state variables
if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = True  # Start in stopped state
if "start_tracking" not in st.session_state:
    st.session_state.start_tracking = False

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1h")  # Fetch last 3 months data with hourly intervals
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error to avoid crashing

# Function to calculate technical indicators
def add_technical_indicators(df):
    if df.empty:
        return df  # Return empty DataFrame if input is empty
    try:
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
        df["BB_High"], df["BB_Mid"], df["BB_Low"] = ta.volatility.bollinger_hband(df["Close"]), ta.volatility.bollinger_mavg(df["Close"]), ta.volatility.bollinger_lband(df["Close"])
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

# Function to get market sentiment from financial news
def get_market_sentiment(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
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
            st.warning(f"Failed to fetch news for {ticker}. Status code: {response.status_code}")
            return 0  # Neutral sentiment if unable to fetch
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
        return 0

# Function to generate buy/sell signals
def generate_signals(df):
    if df.empty:
        return df
    try:
        # Initialize lists with the same length as df
        buy_signals = [None] * len(df)
        sell_signals = [None] * len(df)
        buy_prices = [None] * len(df)  # Store buy prices
        sell_prices = [None] * len(df)  # Store sell prices

        for i in range(1, len(df)):  # Avoid index errors at i=0
            if "RSI" in df and "MACD" in df and "MACD_Signal" in df:
                if df["RSI"].iloc[i] < 30 and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]:
                    buy_signals[i] = 1 # df["Close"].iloc[i]
                    buy_prices[i] = df["Close"].iloc[i]
                elif df["RSI"].iloc[i] > 70 and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]:
                    sell_signals[i] = 1 #df["Close"].iloc[i]
                    sell_prices[i] = df["Close"].iloc[i]
            else:
                st.error("RSI, MACD, or MACD_Signal columns not found in DataFrame.")
                return df

        # Create new columns directly.
        df["Buy_Signal"] = pd.Series(buy_signals, index=df.index)
        df["Sell_Signal"] = pd.Series(sell_signals, index=df.index)
        df["Buy_Price"] = pd.Series(buy_prices, index=df.index) #store prices
        df["Sell_Price"] = pd.Series(sell_prices, index=df.index)
        return df
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return df



# Streamlit UI
st.title("📈 Real-time Stock Tracker with AI-based Signals")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Buttons for starting and stopping tracking
col1, col2 = st.columns([1, 1])  # Adjust the column widths to bring buttons closer
with col1:
    start_tracking_button = st.button("Start Tracking")
with col2:
    stop_tracking_button = st.button("Stop Tracking")

placeholder = st.empty()  # Placeholder for updating content dynamically

# Update session state based on button clicks
if start_tracking_button:
    st.session_state.stop_tracking = False
    st.session_state.start_tracking = True
    st.rerun()  # Force a rerun to start the loop

if stop_tracking_button:
    st.session_state.stop_tracking = True
    st.session_state.start_tracking = False
    st.rerun()  # Stop loop and show stopped message


# Real-time tracking loop
if not st.session_state.stop_tracking:
    while True:
        df = get_stock_data(ticker)
        if df.empty:
            time.sleep(15)
            continue
        df = add_technical_indicators(df)
        df = generate_signals(df)
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
        if "EMA_20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color='blue')))

        # Bollinger Bands
        if "BB_High" in df and "BB_Low" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], mode="lines", name="Bollinger High", line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], mode="lines", name="Bollinger Low", line=dict(color='red')))

        # Buy Signals (🔵) - Changed to vertical dotted lines
        if "Buy_Signal" in df:
            buy_signal_data = df[df["Buy_Signal"].notnull()]
            if not buy_signal_data.empty:
                for index, row in buy_signal_data.iterrows():
                    fig.add_shape(
                        type="line",
                        x0=index,
                        x1=index,
                        y0=df['Low'].min(),
                        y1=df['High'].max(),
                        line=dict(
                            color="blue",
                            width=2,
                            dash="dot",
                        ),
                        name="Buy Signal",
                    )
                    # Display Buy Price
                    fig.add_annotation(
                        x=index,
                        y=df['High'].max(),  # Position above the line
                        text=f"Buy: {row['Buy_Price']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color="blue"),
                    )

        # Sell Signals (🔴) - Changed to vertical dotted lines
        if "Sell_Signal" in df:
            sell_signal_data = df[df["Sell_Signal"].notnull()]
            if not sell_signal_data.empty:
                for index, row in sell_signal_data.iterrows():
                    fig.add_shape(
                        type="line",
                        x0=index,
                        x1=index,
                        y0=df['Low'].min(),
                        y1=df['High'].max(),
                        line=dict(
                            color="red",
                            width=2,
                            dash="dot",
                        ),
                        name="Sell Signal",
                    )
                    # Display Sell Price
                    fig.add_annotation(
                        x=index,
                        y=df['High'].max(),  # Position above the line
                        text=f"Sell: {row['Sell_Price']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color="red"),
                    )

        # Update layout
        fig.update_layout(title=f"{ticker} Stock Performance", xaxis_rangeslider_visible=False)

        # Display chart and signals
        with placeholder.container():
            st.plotly_chart(fig, key=f"chart_{time.time()}")
            st.write(f"**Market Sentiment Score:** {sentiment} (Higher is better)")

        time.sleep(15)
        st.rerun()
