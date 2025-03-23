import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import time
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize session state variables
if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = True
if "start_tracking" not in st.session_state:
    st.session_state.start_tracking = False

# Function to fetch stock data
def get_stock_data(ticker, period="1y", interval="1h"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to calculate technical indicators
def add_technical_indicators(df):
    if df.empty:
        return df
    try:
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
        df["BB_High"], df["BB_Mid"], df["BB_Low"] = ta.volatility.bollinger_hband(df["Close"]), ta.volatility.bollinger_mavg(df["Close"]), ta.volatility.bollinger_lband(df["Close"])
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
        df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
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
            for h in headlines[:5]:
                text = h.text.lower()
                if any(word in text for word in ["rises", "soars", "strong", "bullish", "positive"]):
                    sentiment_score += 1
                elif any(word in text for word in ["drops", "falls", "weak", "bearish", "negative"]):
                    sentiment_score -= 1
            return sentiment_score
        else:
            st.warning(f"Failed to fetch news for {ticker}. Status code: {response.status_code}")
            return 0
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
        return 0

# Function to generate buy/sell signals
def generate_signals(df):
    if df.empty:
        return df
    try:
        buy_signals = [None] * len(df)
        sell_signals = [None] * len(df)
        buy_prices = [None] * len(df)
        sell_prices = [None] * len(df)

        for i in range(1, len(df)):
            if "RSI" in df and "MACD" in df and "MACD_Signal" in df and "ADX" in df and "VWAP" in df:
                if (
                    df["RSI"].iloc[i] < 30
                    and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]
                    and df["ADX"].iloc[i] > 25
                    and df["Close"].iloc[i] > df["VWAP"].iloc[i]
                ):
                    buy_signals[i] = 1
                    buy_prices[i] = df["Close"].iloc[i]
                elif (
                    df["RSI"].iloc[i] > 70
                    and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]
                    and df["ADX"].iloc[i] > 25
                    and df["Close"].iloc[i] < df["VWAP"].iloc[i]
                ):
                    sell_signals[i] = 1
                    sell_prices[i] = df["Close"].iloc[i]

        df["Buy_Signal"] = pd.Series(buy_signals, index=df.index)
        df["Sell_Signal"] = pd.Series(sell_signals, index=df.index)
        df["Buy_Price"] = pd.Series(buy_prices, index=df.index)
        df["Sell_Price"] = pd.Series(sell_prices, index=df.index)
        return df
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return df

# Streamlit UI
st.title("📈 Real-time Stock Tracker with AI-based Signals")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Buttons for starting and stopping tracking
col1, col2 = st.columns([1, 1])
with col1:
    start_tracking_button = st.button("Start Tracking")
with col2:
    stop_tracking_button = st.button("Stop Tracking")

# Sidebar for explanations
with st.sidebar:
    explanation_md = """
        **Explanation of Terms:**

        **Market Sentiment Score:**
        * This score indicates the general mood of the market towards the selected stock, based on an analysis of recent news headlines.
        * The score is calculated by counting positive and negative keywords in the news.
            * A higher score suggests more positive sentiment.
            * A lower or negative score suggests more negative sentiment.
        * It is a general indicator and should be used with other information, not as a sole predictor of buy/sell decisions.

        **Technical Indicators:**
        * **EMA (Exponential Moving Average):** Averages prices, giving more weight to recent data.  Helps identify trends.
        * **RSI (Relative Strength Index):** Measures how overbought or oversold a stock is, suggesting potential reversals.
        * **MACD (Moving Average Convergence Divergence):** Shows the relationship between two moving averages and can signal trend changes.
        * **Bollinger Bands:** Show the range where a stock's price typically trades, with high/low bands marking potential extremes.
        * **ADX (Average Directional Index):** Measures the strength of a trend (not whether it's up or down).
        * **VWAP (Volume Weighted Average Price):** The average price of a stock over a trading day, weighted by volume.

        **Buy/Sell Signals:**
        * The app generates buy/sell signals based on a combination of technical indicators:
            * **Buy Signal:**
                * RSI is less than 30 (oversold).
                * MACD is above its signal line (bullish momentum).
                * ADX is greater than 25 (strong trend).
                * Close price is above VWAP.
            * **Sell Signal:**
                * RSI is greater than 70 (overbought).
                * MACD is below its signal line (bearish momentum).
                * ADX is greater than 25 (strong trend).
                * Close price is below VWAP.
        * These signals suggest potential trading opportunities, but should be used with caution and combined with other analysis.

        **Disclaimer:**
        * The buy/sell signals generated by this app are based on technical analysis and market sentiment.
        * They are not financial advice and should not be used as the sole basis for making investment decisions.
        * Past performance is not indicative of future results.
        * The stock market involves risk, and you could lose money.
        * Consult with a qualified financial advisor before making any investment decisions.
        """
    st.markdown(explanation_md)

placeholder = st.empty()

# Update session state based on button clicks
if start_tracking_button:
    st.session_state.stop_tracking = False
    st.session_state.start_tracking = True
    st.rerun()

if stop_tracking_button:
    st.session_state.stop_tracking = True
    st.session_state.start_tracking = False
    st.rerun()

# Real-time tracking loop
if not st.session_state.stop_tracking:
    df = get_stock_data(ticker, period="1y", interval="1h")
    if df.empty:
        time.sleep(15)
        st.rerun()

    df = add_technical_indicators(df)

    if df.empty:
        st.error(f"No data available for {ticker}. Please check the ticker symbol and try again.")
        time.sleep(15)
        st.rerun()

    df = generate_signals(df)
    sentiment = get_market_sentiment(ticker)

    # Debugging: Print indicator values
    if not df.empty:
        st.write(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
        st.write(f"Current MACD: {df['MACD'].iloc[-1]:.2f}, Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
        st.write(f"Current ADX: {df['ADX'].iloc[-1]:.2f}")
        st.write(f"Current Close: {df['Close'].iloc[-1]:.2f}, VWAP: {df['VWAP'].iloc[-1]:.2f}")

        # Check for buy/sell signals and display them
        if df["Buy_Signal"].iloc[-1] == 1:
            st.write(f"AI Recommendation: Buy {ticker} at {df['Close'].iloc[-1]:.2f}")
        elif df["Sell_Signal"].iloc[-1] == 1:
            st.write(f"AI Recommendation: Sell {ticker} at {df['Close'].iloc[-1]:.2f}")
        else:
            st.write(f"AI Recommendation: No Action on {ticker}")

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

    if "VWAP" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines", name="VWAP", line=dict(color='purple')))

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
                    y=df['High'].max(),
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
                    y=df['High'].max(),
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
