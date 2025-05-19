import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta

# ----------------- App Setup ------------------
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Real-Time Trading Dashboard")

# Sidebar Inputs
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "1d"])
show_sma = st.sidebar.checkbox("Show SMA 200", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_fvg = st.sidebar.checkbox("Show Fair Value Gaps", value=True)

# ----------------- Data Fetching ------------------
interval_map = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"
}
period = "7d" if timeframe in ["1m", "5m", "15m", "1h"] else "1mo"

df = yf.download(tickers=symbol, interval=interval_map[timeframe], period=period)

if df.empty:
    st.error("⚠️ No data returned. Try a different symbol or timeframe.")
    st.stop()

# Reset index & normalize columns
df.reset_index(inplace=True)
df.columns = [str(col).strip().title() for col in df.columns]

# Rename Date to Datetime if needed
if "Date" in df.columns:
    df.rename(columns={"Date": "Datetime"}, inplace=True)

# Print available columns
required_cols = ["Open", "High", "Low", "Close", "Volume", "Datetime"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing expected columns: {missing_cols}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# ----------------- Indicators ------------------
if show_sma:
    try:
        df["Sma200"] = ta.trend.sma_indicator(df["Close"], window=200)
    except Exception as e:
        st.warning(f"⚠️ Could not compute SMA: {e}")

if show_macd:
    try:
        macd = ta.trend.MACD(df["Close"])
        df["Macd"] = macd.macd()
        df["Macd_Signal"] = macd.macd_signal()
        df["Macd_Hist"] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Could not compute MACD: {e}")

if show_rsi:
    try:
        df["Rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    except Exception as e:
        st.warning(f"⚠️ Could not compute RSI: {e}")

# ----------------- Plotting ------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["Datetime"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Price"
))

if show_sma and "Sma200" in df:
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["Sma200"], name="SMA 200",
        line=dict(color="blue", width=1.5)
    ))

if show_fvg:
    for i in range(2, len(df)):
        a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        if abs(a["Close"] - a["Open"]) > abs(b["Close"] - b["Open"]) * 2:
            fvg_start = a["Close"]
            fvg_end = c["Open"]
            fig.add_shape(
                type="rect", x0=a["Datetime"], x1=c["Datetime"],
                y0=min(fvg_start, fvg_end), y1=max(fvg_start, fvg_end),
                fillcolor="orange", opacity=0.2, line_width=0, layer="below"
            )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if theme == "Dark" else "plotly_white",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- RSI Plot ------------------
if show_rsi and "Rsi" in df:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Datetime"], y=df["Rsi"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig_rsi.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=300
    )
    st.subheader("RSI (Relative Strength Index)")
    st.plotly_chart(fig_rsi, use_container_width=True)

# ----------------- Volume Plot ------------------
if show_volume and "Volume" in df:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], name="Volume", marker_color="gray"))
    fig_vol.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=250
    )
    st.subheader("Volume")
    st.plotly_chart(fig_vol, use_container_width=True)
