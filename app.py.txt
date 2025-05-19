import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import ta

# Streamlit layout setup
st.set_page_config(layout="wide", page_title="Trading Chart with Indicators")

# Sidebar UI
st.sidebar.title("📈 Chart Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
dark_mode = st.sidebar.checkbox("Dark Mode", value=True)

show_sma = st.sidebar.checkbox("Show SMA 200", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_fvg = st.sidebar.checkbox("Show Fair Value Gaps", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

# Mapping for yfinance intervals
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}

# Fetch historical data
df = yf.download(symbol, period="7d", interval=interval_map[timeframe])
df.dropna(inplace=True)

# Technical Indicators
df["SMA200"] = ta.trend.sma_indicator(df["Close"], window=200)
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()
df["MACD_Hist"] = macd.macd_diff()
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

# Fair Value Gap logic
def find_fvgs(data):
    zones = []
    for i in range(2, len(data)):
        a, b, c = data.iloc[i - 2], data.iloc[i - 1], data.iloc[i]
        if (abs(a["Close"] - a["Open"]) > (a["High"] - a["Low"]) * 0.6 and
            abs(b["Close"] - b["Open"]) < (b["High"] - b["Low"]) * 0.2 and
            abs(c["Open"] - a["Close"]) > (a["High"] - a["Low"]) * 0.3):
            zones.append((data.index[i], a["Close"], c["Open"]))
    return zones

fvg_zones = find_fvgs(df)

# Plotly Chart
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Candles"
))

# SMA 200
if show_sma:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA200"],
        line=dict(color="orange", width=2),
        name="SMA 200"
    ))

# FVG Zones
if show_fvg:
    for ts, y1, y2 in fvg_zones:
        fig.add_shape(type="rect", x0=ts, x1=ts, y0=min(y1, y2), y1=max(y1, y2),
                      line=dict(width=0), fillcolor="rgba(255,0,0,0.2)", layer="below")

# Volume Bars
if show_volume:
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], yaxis="y2", name="Volume", marker_color="blue", opacity=0.3
    ))

# Layout settings
fig.update_layout(
    title=f"{symbol} Price Chart",
    xaxis_title="Time",
    yaxis_title="Price",
    yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1),
    template="plotly_dark" if dark_mode else "plotly_white",
    height=600
)

# Main price chart
st.plotly_chart(fig, use_container_width=True)

# MACD Chart
if show_macd:
    st.subheader("MACD")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="cyan")))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color="magenta")))
    macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="gray"))
    macd_fig.update_layout(height=300, template="plotly_dark" if dark_mode else "plotly_white")
    st.plotly_chart(macd_fig, use_container_width=True)

# RSI Chart
if show_rsi:
    st.subheader("RSI")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="yellow"), name="RSI"))
    rsi_fig.add_hline(y=70, line=dict(color="red", dash="dot"))
    rsi_fig.add_hline(y=30, line=dict(color="green", dash="dot"))
    rsi_fig.update_layout(height=300, template="plotly_dark" if dark_mode else "plotly_white")
    st.plotly_chart(rsi_fig, use_container_width=True)
