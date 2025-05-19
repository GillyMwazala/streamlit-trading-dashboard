import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta

# ---------------------------------------------
# UI Config
# ---------------------------------------------
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Real-Time Trading Dashboard")

# Light/Dark mode toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

# Symbol & Timeframe selectors
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "1d"])

# Indicator toggles
show_sma = st.sidebar.checkbox("Show SMA 200", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_fvg = st.sidebar.checkbox("Show Fair Value Gaps", value=True)

# ---------------------------------------------
# Data Fetching
# ---------------------------------------------
interval_map = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "1d": "1d"
}
period = "7d" if timeframe in ["1m", "5m", "15m", "1h"] else "1mo"

st.subheader(f"📈 {symbol} - {timeframe} Chart")

df = yf.download(tickers=symbol, interval=interval_map[timeframe], period=period)

# Check for valid data
# First, check if the DataFrame is empty
if df.empty:
    st.error("❌ Failed to fetch data (DataFrame is empty). Try a different symbol or timeframe.")
    st.stop()

# Reset the index to make the Datetime index a column
df.reset_index(inplace=True)

# Normalize all column names (including the one from reset_index)
normalized_columns = []
for col in df.columns:
    if isinstance(col, tuple):
        # For tuples like ('Close', 'BTC-USD') -> 'Close'
        # For ('', 'Datetime') -> 'Datetime' (if first part is empty)
        # For ('Datetime', '') -> 'Datetime'
        name_parts = [part for part in col if part]  # Get non-empty parts of the tuple
        name = name_parts[0] if name_parts else str(col)  # Take first non-empty, or full tuple as string if all empty
        normalized_columns.append(name.strip().title())
    else: # Non-tuple column name
        col_name_str = str(col).strip().title()
        # If reset_index created a column named 'Index' (original index was unnamed),
        # and we haven't already processed/added a 'Datetime' column:
        if col_name_str == 'Index' and 'Datetime' not in normalized_columns:
            normalized_columns.append('Datetime')  # Assume this 'Index' column is our time axis
        else:
            # Keep other names like 'Open', 'Close', or an existing 'Datetime'
            normalized_columns.append(col_name_str)
df.columns = normalized_columns

# Now, check if all required columns are present after normalization
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']
missing_cols = [rc for rc in required_cols if rc not in df.columns]
if missing_cols:
    st.error(f"❌ Data fetched, but missing essential columns after normalization: {', '.join(missing_cols)}.\nAvailable columns: {', '.join(df.columns)}")
    st.stop()

# ---------------------------------------------
# Indicators
# ---------------------------------------------
if show_sma and "Close" in df:
    df["Sma200"] = ta.trend.sma_indicator(df["Close"], window=200)

if show_macd and "Close" in df:
    macd = ta.trend.MACD(df["Close"])
    df["Macd"] = macd.macd()
    df["Macd_Signal"] = macd.macd_signal()
    df["Macd_Hist"] = macd.macd_diff()

if show_rsi and "Close" in df:
    df["Rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

# ---------------------------------------------
# Plotly Chart
# ---------------------------------------------
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df["Datetime"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))

# SMA 200
if show_sma and "Sma200" in df:
    fig.add_trace(go.Scatter(
        x=df["Datetime"],
        y=df["Sma200"],
        line=dict(color="blue", width=1.5),
        name="SMA 200"
    ))

# Fair Value Gaps
if show_fvg:
    for i in range(2, len(df)):
        a = df.iloc[i - 2]
        b = df.iloc[i - 1]
        c = df.iloc[i]

        body_a = abs(a["Close"] - a["Open"])
        body_b = abs(b["Close"] - b["Open"])

        if body_a > body_b * 2 and (c["Open"] > a["Close"] or c["Open"] < a["Close"]):
            fvg_start = a["Close"]
            fvg_end = c["Open"]
            fig.add_shape(
                type="rect",
                x0=a["Datetime"],
                x1=c["Datetime"],
                y0=min(fvg_start, fvg_end),
                y1=max(fvg_start, fvg_end),
                fillcolor="orange",
                opacity=0.2,
                line_width=0,
                layer="below"
            )

# Chart Layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if theme == "Dark" else "plotly_white",
    margin=dict(l=0, r=0, t=30, b=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# RSI Panel
# ---------------------------------------------
if show_rsi and "Rsi" in df:
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Datetime"], y=df["Rsi"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig_rsi.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------------------------------
# Volume Chart
# ---------------------------------------------
if show_volume and "Volume" in df:
    st.subheader("Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], name="Volume", marker_color="gray"))
    fig_vol.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_vol, use_container_width=True)
