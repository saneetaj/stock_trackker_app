import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
def get_stock_data(ticker, period="1y", interval="1h"):  # Increased period for optimization
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
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])  # Added ADX
        df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"]) # Added VWAP
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
def generate_signals(df, rsi_window, macd_fast, macd_slow, macd_signal_window):  # Added parameters
    if df.empty:
        return df
    try:
        # Calculate indicators with custom parameters
        df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_window)
        df["MACD"] = ta.trend.macd(df["Close"], window_fast=macd_fast, window_slow=macd_slow)
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal_window)
        
        buy_signals = [None] * len(df)
        sell_signals = [None] * len(df)
        buy_prices = [None] * len(df)  # Store buy prices
        sell_prices = [None] * len(df)  # Store sell prices

        for i in range(1, len(df)):
            if "RSI" in df and "MACD" in df and "MACD_Signal" in df and "ADX" in df and "VWAP" in df: #check for the new columns
                if (
                    df["RSI"].iloc[i] < 30
                    and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]
                    and df["ADX"].iloc[i] > 25  # ADX filter: trend strength
                    and df["Close"].iloc[i] > df["VWAP"].iloc[i] # Price above VWAP
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
            else:
                st.error("Required columns not found in DataFrame.")
                return pd.DataFrame() # Return empty df

        df["Buy_Signal"] = pd.Series(buy_signals, index=df.index)
        df["Sell_Signal"] = pd.Series(sell_signals, index=df.index)
        df["Buy_Price"] = pd.Series(buy_prices, index=df.index)
        df["Sell_Price"] = pd.Series(sell_prices, index=df.index)
        return df
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return df

# Function for backtesting
def backtest(df, rsi_window, macd_fast, macd_slow, macd_signal_window, initial_capital=10000):
    try:
        if df.empty or not all(col in df for col in ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_Signal", "ADX", "VWAP", "Buy_Signal", "Sell_Signal"]):
            raise ValueError("Invalid DataFrame: Missing data or empty, or missing required columns.")

        
        positions = []
        capital = initial_capital
        buy_price = 0  # Initialize buy_price here to avoid NameError.  This is CRUCIAL
        
        for i in range(len(df)):
            if df["Buy_Signal"].iloc[i] == 1 and not positions:
                positions.append(("Buy", df["Close"].iloc[i], capital / df["Close"].iloc[i], i))  # (Buy/Sell, Price, Quantity, Index)
                capital = 0
                buy_price = df["Close"].iloc[i] # Capture buy price
            elif df["Sell_Signal"].iloc[i] == 1 and positions:
                quantity = positions[0][2]
                capital = quantity * df["Close"].iloc[i]
                positions.append(("Sell", df["Close"].iloc[i], quantity, i))
                positions = []  # Clear positions after selling

        if positions: # Handle any open positions at the end
            final_price = df["Close"].iloc[-1]
            capital = positions[0][2] * final_price
            positions.append(("Sell", final_price, positions[0][2], len(df)-1))

        total_profit = capital - initial_capital
        
        # Calculate max drawdown
        peak = initial_capital
        drawdown = 0
        max_drawdown = 0
        for p in positions:
            if p[0] == "Buy":
                peak = initial_capital
            elif p[0] == "Sell":
                peak = max(peak, capital + (p[1] - buy_price) * p[2])
                drawdown = peak - capital
                max_drawdown = max(max_drawdown, drawdown)
        
        
        if initial_capital > 0:
          profit_factor = capital / initial_capital if initial_capital > 0 else 0
        else:
          profit_factor = 0
        return total_profit, profit_factor, max_drawdown, positions #returning the positions
    except Exception as e:
        st.error(f"Error in backtest: {e} with parameters (RSI={rsi_window}, MACD Fast={macd_fast}, MACD Slow={macd_slow}, Signal={macd_signal_window})")
        return 0, 0, 0, []


# Function to optimize parameters
def optimize_parameters(df):
    best_profit_factor = -float('inf')
    best_rsi_window = None
    best_macd_fast = None
    best_macd_slow = None
    best_macd_signal_window = None
    
    rsi_windows = [10, 14, 20]
    macd_fasts = [12]
    macd_slows = [26]
    macd_signal_windows = [9]
    
    if df is None or df.empty:
        st.error("Error: optimize_parameters received an empty DataFrame.")
        return None, None, None, None, 0
    
    if not isinstance(df, pd.DataFrame):
        st.error("Error: optimize_parameters received data that is not a DataFrame.")
        return None, None, None, None, 0
    
    for rsi_window in rsi_windows:
        for macd_fast in macd_fasts:
            for macd_slow in macd_slows:
                for macd_signal_window in macd_signal_windows:
                    try:
                        # Before calling backtest, check if df is valid and contains essential columns
                        if df.empty or not all(col in df for col in ["Open", "High", "Low", "Close", "Volume"]):
                            st.error(f"Invalid DataFrame before backtest with parameters (RSI={rsi_window}, MACD Fast={macd_fast}, MACD Slow={macd_slow}, Signal={macd_signal_window})")
                            return None, None, None, None, 0  # Return a default value

                        # Create a copy here to avoid modifying the original DataFrame
                        df_copy = df.copy()

                        # Add the technical indicators here, before backtesting
                        df_copy = add_technical_indicators(df_copy)
                        
                        # Check again, after adding indicators, if the df_copy is valid for backtest
                        if df_copy.empty or not all(col in df_copy for col in ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_Signal", "ADX", "VWAP"]):
                            st.error(f"Invalid DataFrame before backtest with parameters (RSI={rsi_window}, MACD Fast={macd_fast}, MACD Slow={macd_slow}, Signal={macd_signal_window})")
                            return None, None, None, None, 0  # Return a default value

                        profit, profit_factor, max_drawdown, positions = backtest(df_copy, rsi_window, macd_fast, macd_slow, macd_signal_window)
                        if profit_factor > best_profit_factor:
                            best_profit_factor = profit_factor
                            best_rsi_window = rsi_window
                            best_macd_fast = macd_fast
                            best_macd_slow = macd_slow
                            best_macd_signal_window = macd_signal_window
                    except Exception as e:
                        st.error(f"Error in backtest with parameters (RSI={rsi_window}, MACD Fast={macd_fast}, MACD Slow={macd_slow}, Signal={macd_signal_window}): {e}")
                        return None, None, None, None, 0 # Return default
    return best_rsi_window, best_macd_fast, best_macd_slow, best_macd_signal_window, best_profit_factor

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
        * This score indicates the general mood of the market towards the selected stock, based on analysis of recent news headlines.
        * The score is calculated by counting positive and negative keywords in the news.
            * A higher score suggests more positive sentiment.
            * A lower or negative score suggests more negative sentiment.
        * It's a general indicator and should be used with other information, not as a sole predictor of buy/sell decisions.

        **Optimized Parameters:**
        * The app uses technical indicators (RSI and MACD) to generate buy/sell signals. These indicators have adjustable settings (parameters).
        * The app automatically tries different parameter values to find the combination that historically would have produced the best results.
        * RSI Window: The number of past periods used to calculate the Relative Strength Index (RSI).
        * MACD Fast: The shorter period EMA used in the MACD calculation.
        * MACD Slow: The longer period EMA used in the MACD calculation.
        * MACD Signal: The number of periods used to calculate the signal line of the MACD.
        * Optimized Profit Factor: The profit factor achieved by the optimized parameters in historical testing.

        **Backtesting Results:**
        * Backtesting is the process of testing a trading strategy on historical data to see how it would have performed.
        * The app performs a simplified backtest to evaluate the effectiveness of the buy/sell signals generated with the optimized parameters.
        * Total Profit: The total profit or loss that would have been generated by the strategy over the backtesting period, in dollars.
        * Profit Factor: A measure of a trading strategy's profitability. It's calculated as the ratio of gross profit to gross loss.
            * A profit factor greater than 1 indicates a profitable strategy.
            * A higher profit factor is generally better.
        * Max Drawdown: The largest peak-to-trough decline during the backtesting period. It indicates the potential risk of the strategy.
            * A lower max drawdown is generally better, as it indicates less risk.
        * Trades: The individual trades that the backtest would have made, including the type of trade (buy or sell), the price, and the date.

        **How this Explains Buy/Sell Recommendations:**
        * The app uses the optimized parameters in its core buy/sell signal generation logic. The backtesting results show how those parameters would have performed historically. So, the parameters are chosen to try to maximize profitability, and the backtest shows you how that worked out. The market sentiment is an additional piece of information to consider.
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
    
    if df.empty: #check if the dataframe is empty
        st.error(f"No data available for {ticker}. Please check the ticker symbol and try again.")
        time.sleep(15)
        st.rerun()
    
    # Optimize parameters
    best_rsi_window, best_macd_fast, best_macd_slow, best_macd_signal_window, profit_factor = optimize_parameters(df.copy())
    
    # Check if optimize_parameters returned valid values
    if best_rsi_window is not None and best_macd_fast is not None and best_macd_slow is not None and best_macd_signal_window is not None:
        
        df = generate_signals(df, best_rsi_window, best_macd_fast, best_macd_slow, best_macd_signal_window)
        sentiment = get_market_sentiment(ticker)

        # Debugging: Print indicator values
        if not df.empty:
            st.write(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
            st.write(f"Current MACD: {df['MACD'].iloc[-1]:.2f}, Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
            st.write(f"Current ADX: {df['ADX'].iloc[-1]:.2f}")
            st.write(f"Current Close: {df['Close'].iloc[-1]:.2f}, VWAP: {df['VWAP'].iloc[-1]:.2f}")

            # Check for buy/sell signals and display them
            if df["Buy_Signal"].iloc[-1] == 1:
                st.write(f"Buy Signal for {ticker} at {df['Close'].iloc[-1]:.2f}")
            elif df["Sell_Signal"].iloc[-1] == 1:
                st.write(f"Sell Signal for {ticker} at {df['Close'].iloc[-1]:.2f}")
            else:
                st.write(f"No Buy/Sell Signal for {ticker}")

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
            fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines", name="VWAP", line=dict(color='purple'))) #add vwap

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

            # Backtest and display results
            # Create a copy here to avoid modifying the original DataFrame
            df_copy_backtest = df.copy()
            profit, profit_factor, max_drawdown, positions = backtest(df_copy_backtest, best_rsi_window, best_macd_fast, best_macd_slow, best_macd_signal_window) # Pass the parameters to backtest
            st.write("Backtesting Results:")
            st.write(f"  Total Profit: {profit:.2f}")
            st.write(f"  Profit Factor: {profit_factor:.2f}")
            st.write(f"  Max Drawdown: {max_drawdown:.2f}")
            st.write("  Trades:")
            for trade in positions:
                st.write(f"    {trade[0]} at {trade[1]:.2f} on {df.index[trade[3]]}")

        # Update layout
        fig.update_layout(title=f"{ticker} Stock Performance", xaxis_rangeslider_visible=False)

        # Display chart and signals
        with placeholder.container():
            st.plotly_chart(fig, key=f"chart_{time.time()}")
            st.write(f"**Market Sentiment Score:** {sentiment} (Higher is better)")

            time.sleep(15)
            st.rerun()
    else:
        st.error("Failed to obtain optimized parameters. Please check the data and try again.")
        time.sleep(15)
        st.rerun()
