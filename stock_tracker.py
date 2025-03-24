import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import time
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai

if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = True
if "start_tracking" not in st.session_state:
    st.session_state.start_tracking = False
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = ["EMA_20", "RSI", "MACD", "BB_High", "BB_Mid", "BB_Low", "ADX", "VWAP"]
if "tracked_tickers" not in st.session_state:
    st.session_state.tracked_tickers = ["AAPL"]

GENAI_KEY = "AIzaSyCxxmnu1LIdKD4hVkTuprO274ww5hv4R5k"
genai.configure(api_key=GENAI_KEY)
model = genai.GenerativeModel("gemini-pro")


def get_stock_data(ticker, period="1y", interval="1h"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def add_technical_indicators(df):
    if df.empty:
        return df
    try:
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
        df["BB_High"], df["BB_Mid"], df["BB_Low"] = ta.volatility.bollinger_hband(df["Close"]), ta.volatility.bollinger_mavg(
            df["Close"]), ta.volatility.bollinger_lband(df["Close"])
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
        df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df



def get_market_sentiment_gemini(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            headlines = soup.find_all("h3")
            news_text = " ".join([h.text for h in headlines[:5]])

            prompt = (
                "Analyze the market sentiment for " + ticker + " by considering the following:\n\n"
                "1.  News Headlines: \"" + news_text + "\"\n"
                "2.  General web sentiment from financial blogs, forums, and other sources.\n\n"
                "Provide a concise summary of the overall market sentiment, categorizing it as positive, negative, or neutral.  Also, include the key reasons\n"
                "influencing the sentiment."
            )
            gemini_response = model.generate_content(prompt)
            return gemini_response.text

        else:
            st.warning(
                f"Failed to fetch news for {ticker}. Status code: {response.status_code}")
            return "N/A"
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
        return "N/A"



def generate_signals(df):
    if df.empty:
        return df
    try:
        buy_signals = [None] * len(df)
        sell_signals = [None] * len(df)
        buy_prices = [None] * len(df)
        sell_prices = [None] * len(df)
        buy_reasons = [None] * len(df)
        sell_reasons = [None] * len(df)

        for i in range(1, len(df)):
            reasons = []
            if "RSI" in df and "MACD" in df and "MACD_Signal" in df and "ADX" in df and "VWAP" in df:
                if (
                        df["RSI"].iloc[i] < 30
                        and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]
                        and df["ADX"].iloc[i] > 25
                        and df["Close"].iloc[i] > df["VWAP"].iloc[i]
                ):
                    buy_signals[i] = 1
                    buy_prices[i] = df["Close"].iloc[i]
                    reasons.append("RSI < 30 (Oversold)")
                    reasons.append("MACD > Signal (Bullish)")
                    reasons.append("ADX > 25 (Strong Trend)")
                    reasons.append("Close > VWAP")
                    buy_reasons[i] = ", ".join(reasons)
                elif (
                        df["RSI"].iloc[i] > 70
                        and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]
                        and df["ADX"].iloc[i] > 25
                        and df["Close"].iloc[i] < df["VWAP"].iloc[i]
                ):
                    sell_signals[i] = 1
                    sell_prices[i] = df["Close"].iloc[i]
                    reasons.append("RSI > 70 (Overbought)")
                    reasons.append("MACD < Signal (Bearish)")
                    reasons.append("ADX > 25 (Strong Trend)")
                    reasons.append("Close < VWAP")
                    sell_reasons[i] = ", ".join(reasons)
            else:
                st.error("Required columns not found in DataFrame.")
                return df
        df["Buy_Signal"] = pd.Series(buy_signals, index=df.index)
        df["Sell_Signal"] = pd.Series(sell_signals, index=df.index)
        df["Buy_Price"] = pd.Series(buy_prices, index=df.index)
        df["Sell_Price"] = pd.Series(sell_prices, index=df.index)
        df["Buy_Reasons"] = pd.Series(buy_reasons, index=df.index)
        df["Sell_Reasons"] = pd.Series(sell_reasons, index=df.index)
        return df
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return df



def get_analyst_ratings(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/analysis"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            target_est = soup.find("td", {"class": "Ta(end) Fw(b) Lh(1.42)"}).text if soup.find(
                "td", {"class": "Ta(end) Fw(b) Lh(1.42)"}) else "N/A"
            rating = soup.find("div", {"class": "Fw(600) Mt(8px) D(ib)"}).text if soup.find(
                "div", {"class": "Fw(600) Mt(8px) D(ib)"}) else "N/A"
            return target_est, rating
        else:
            st.warning(f"Failed to fetch analyst ratings for {ticker}. Status code: {response.status_code}")
            return "N/A", "N/A"
    except Exception as e:
        st.error(f"Error fetching analyst ratings: {e}")
        return "N/A", "N/A"



st.title("📈 Real-time Stock Tracker with AI-powered Analysis")

tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, MSFT):", "AAPL,TSLA,MSFT")
tickers_list = [ticker.strip() for ticker in tickers.split(",")]

col1, col2 = st.columns([1, 1])
with col1:
    start_tracking_button = st.button("Start Tracking")
with col2:
    stop_tracking_button = st.button("Stop Tracking")

with st.sidebar:
    st.sidebar.subheader("Select Technical Indicators")
    ema_20_selected = st.sidebar.checkbox("EMA 20", value=("EMA_20" in st.session_state.selected_indicators),
                                         key="ema_20_checkbox")
    rsi_selected = st.sidebar.checkbox("RSI", value=("RSI" in st.session_state.selected_indicators), key="rsi_checkbox")
    macd_selected = st.sidebar.checkbox("MACD", value=("MACD" in st.session_state.selected_indicators), key="macd_checkbox")
    bb_selected = st.sidebar.checkbox("Bollinger Bands",
                                     value=("BB_High" in st.session_state.selected_indicators or "BB_Mid" in st.session_state.selected_indicators or "BB_Low" in st.session_state.selected_indicators),
                                     key="bb_checkbox")
    adx_selected = st.sidebar.checkbox("ADX", value=("ADX" in st.session_state.selected_indicators), key="adx_checkbox")
    vwap_selected = st.sidebar.checkbox("VWAP", value=("VWAP" in st.session_state.selected_indicators), key="vwap_checkbox")

    explanation_md = """
        **Explanation of Terms:**

        **Market Sentiment:**
        * This is a summary of the overall market mood towards the selected stock.  It is analyzed by an AI model (Gemini) that considers
            information from news headlines, financial blogs, forums, and other web sources.  The sentiment is categorized as positive,
            negative, or neutral, along with key reasons influencing it.

        **Technical Indicators:**
        * **EMA (Exponential Moving Average):** Averages prices, giving more weight to recent data.  Helps identify trends.
        * **RSI (Relative Strength Index):** Measures how overbought or oversold a stock is, suggesting potential reversals.
        * **MACD (Moving Average Convergence Divergence):** Shows the relationship between two moving averages and can signal trend changes.
        * **Bollinger Bands:** Show the range where a stock's price typically trades, with high/low bands marking potential extremes.
        * **ADX (Average Directional Index):** Measures the strength of a trend (not whether it's up or down).
        * **VWAP (Volume Weighted Average Price):** The average price of a stock over a trading day, weighted by volume.

        **Analyst Ratings:**
        * Displays the latest analyst ratings for the stock, including price targets and buy/sell recommendations.

        """
    st.markdown(explanation_md, unsafe_allow_html=True)

placeholder = st.empty()

if start_tracking_button:
    st.session_state.stop_tracking = False
    st.session_state.start_tracking = True
    st.session_state.tracked_tickers = tickers_list
    st.rerun()
elif stop_tracking_button:
    st.session_state.stop_tracking = True
    st.session_state.start_tracking = False
    st.rerun()

if "tracked_tickers" not in st.session_state:
    st.session_state.tracked_tickers = tickers_list

if not st.session_state.stop_tracking:
    all_dfs = {}
    for ticker in st.session_state.tracked_tickers:
        now = datetime.now()
        period = "2d"
        interval = "5m"

        df = get_stock_data(ticker, period=period, interval=interval)
        if df.empty:
            time.sleep(300)
            st.rerun()
        df = add_technical_indicators(df)
        if df.empty:
            st.error(f"No data available for {ticker}. Please check the ticker symbol and try again.")
            time.sleep(300)
            st.rerun()
        df = generate_signals(df)
        all_dfs[ticker] = df

    sentiment_data = {ticker: get_market_sentiment_gemini(ticker) for ticker in
                      st.session_state.tracked_tickers}
    analyst_ratings = {ticker: get_analyst_ratings(ticker) for ticker in st.session_state.tracked_tickers}

    st.session_state.selected_indicators = []
    if ema_20_selected:
        st.session_state.selected_indicators.append("EMA_20")
    if rsi_selected:
        st.session_state.selected_indicators.append("RSI")
    if macd_selected:
        st.session_state.selected_indicators.append("MACD")
    if bb_selected:
        st.session_state.selected_indicators.extend(["BB_High", "BB_Mid", "BB_Low"])
    if adx_selected:
        st.session_state.selected_indicators.append("ADX")
    if vwap_selected:
        st.session_state.selected_indicators.append("VWAP")

    with placeholder.container():
        for ticker in st.session_state.tracked_tickers:
            df = all_dfs[ticker]
            st.subheader(f"Stock: {ticker}")
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Candlesticks"
            ))

            for indicator in st.session_state.selected_indicators:
                if indicator in df:
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode="lines", name=indicator))

            if "Buy_Signal" in df and "Sell_Signal" in df:
                buy_signal_data = df[df["Buy_Signal"] == 1]
                sell_signal_data = df[df["Sell_Signal"] == 1]

                for index, row in buy_signal_data.iterrows():
                    fig.add_shape(
                        type="line",
                        x0=index,
                        x1=index,
                        y0=df['Low'].min(),
                        y1=df['High'].max(),
                        line=dict(color="blue", width=2, dash="dot"),
                        name="Buy Signal"
                    )
                    fig.add_annotation(
                        x=index,
                        y=df['High'].max(),
                        text=f"Buy: {row['Buy_Price']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color="blue")
                    )

                for index, row in sell_signal_data.iterrows():
                    fig.add_shape(
                        type="line",
                        x0=index,
                        x1=index,
                        y0=df['Low'].min(),
                        y1=df['High'].max(),
                        line=dict(color="red", width=2, dash="dot"),
                        name="Sell Signal"
                    )
                    fig.add_annotation(
                        x=index,
                        y=df['High'].max(),
                        text=f"Sell: {row['Sell_Price']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color="red")
                    )

            fig.update_layout(
                title=f"{ticker} Stock Performance",
                xaxis_rangeslider_visible=False,
                showlegend=True
            )
            st.plotly_chart(fig, key=f"chart_{ticker}_{time.time()}")

            sentiment_text = sentiment_data[ticker]
            target_price, rating = analyst_ratings[ticker]

            st.write(f"**Market Sentiment for {ticker}:**")
            st.write(sentiment_text)
            st.write(f"**Analyst Target Price for {ticker}:** {target_price}")
            st.write(f"**Analyst Rating for {ticker}:** {rating}")

            if not df.empty:
                if df["Buy_Signal"].iloc[-1] == 1:
                    st.write(
                        f"**AI Recommendation for {ticker}:** :green[Buy] at {df['Close'].iloc[-1]:.2f}  **Reasons:** {df['Buy_Reasons'].iloc[-1]}")
                elif df["Sell_Signal"].iloc[-1] == 1:
                    st.write(
                        f"**AI Recommendation for {ticker}:** :red[Sell] at {df['Close'].iloc[-1]:.2f}  **Reasons:** {df['Sell_Reasons'].iloc[-1]}")
                else:
                    st.write(
                        f"**AI Recommendation for {ticker}:** :blue[No Action].   **Reasons:** No strong buy or sell signals detected.")

        time.sleep(300)
        st.rerun()
