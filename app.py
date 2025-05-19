import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# ---------------------------------------------
# UI Config
# ---------------------------------------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📊 Real-Time Trading Dashboard with AI Prediction")

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

# Symbol & Timeframe
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "1d"])

# Toggles
show_sma = st.sidebar.checkbox("Show SMA 200", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_fvg = st.sidebar.checkbox("Show Fair Value Gaps", value=True)
show_ai = st.sidebar.checkbox("AI Price Predictor", value=True)

# ---------------------------------------------
# Data Fetching
# ---------------------------------------------
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
period = "7d" if timeframe in ["1m", "5m", "15m", "1h"] else "3mo"

st.subheader(f"📈 {symbol} - {timeframe} Chart")

df = yf.download(tickers=symbol, interval=interval_map[timeframe], period=period)

if df.empty or "Close" not in df.columns:
    st.error("❌ Failed to fetch data. Try a different symbol or timeframe.")
    st.stop()

df.reset_index(inplace=True)
df.rename(columns={"Date": "Datetime"}, inplace=True)
df.columns = [str(c).title() for c in df.columns]

# ---------------------------------------------
# Indicators
# ---------------------------------------------
if show_sma:
    df["Sma200"] = ta.trend.sma_indicator(df["Close"], window=200)

if show_macd:
    macd = ta.trend.MACD(df["Close"])
    df["Macd"] = macd.macd()
    df["Macd_Signal"] = macd.macd_signal()
    df["Macd_Hist"] = macd.macd_diff()

if show_rsi:
    df["Rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

# ---------------------------------------------
# AI Price Predictor
# ---------------------------------------------
if show_ai:
    st.subheader("🤖 AI Next Close Price Prediction")

    df["Target"] = df["Close"].shift(-1)
    feature_cols = ["Close", "Volume"]

    # Add indicators if available
    if "Sma200" in df: feature_cols.append("Sma200")
    if "Macd" in df: feature_cols.extend(["Macd", "Macd_Signal"])
    if "Rsi" in df: feature_cols.append("Rsi")

    df_model = df[feature_cols + ["Target"]].dropna()

    if len(df_model) > 50:
        X = df_model[feature_cols]
        y = df_model["Target"]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        latest_features = df[feature_cols].iloc[-1:].values
        prediction = model.predict(latest_features)[0]

        st.success(f"📍 Predicted Next Close: **${prediction:.2f}**")
    else:
        st.warning("Not enough data to run AI prediction.")

# ---------------------------------------------
# Candlestick Chart
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

# SMA
if show_sma and "Sma200" in df:
    fig.add_trace(go.Scatter(
        x=df["Datetime"],
        y=df["Sma200"],
        line=dict(color="blue", width=1.5),
        name="SMA 200"
    ))

# FVG Zones
if show_fvg:
    for i in range(2, len(df)):
        a = df.iloc[i - 2]
        b = df.iloc[i - 1]
        c = df.iloc[i]

        body_a = abs(a["Close"] - a["Open"])
        body_b = abs(b["Close"] - b["Open"])

        if body_a > body_b * 2 and (c["Open"] > a["Close"] or c["Open"] < a["Close"]):
            fig.add_shape(
                type="rect",
                x0=a["Datetime"],
                x1=c["Datetime"],
                y0=min(a["Close"], c["Open"]),
                y1=max(a["Close"], c["Open"]),
                fillcolor="orange",
                opacity=0.2,
                line_width=0,
                layer="below"
            )

# Layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if theme == "Dark" else "plotly_white",
    height=600,
    margin=dict(t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# RSI Chart
# ---------------------------------------------
if show_rsi and "Rsi" in df:
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Datetime"], y=df["Rsi"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig_rsi.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=300
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------------------------------
# Volume
# ---------------------------------------------
if show_volume:
    st.subheader("Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], name="Volume", marker_color="gray"))
    fig_vol.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=250
    )
    st.plotly_chart(fig_vol, use_container_width=True)
