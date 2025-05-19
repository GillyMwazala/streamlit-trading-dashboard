# Note: This app requires 'streamlit', 'yfinance', 'pandas', 'plotly', and 'ta' packages.
# Ensure these packages are installed in your environment before running this app.

try:
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
    import ta
except ModuleNotFoundError as e:
    missing_module = str(e).split("No module named '")[1].split("'")[0]
    print(f"Missing required module: {missing_module}. Please install it using pip, e.g., pip install {missing_module}")
    raise SystemExit

# Page setup
st.set_page_config(page_title="📈 Enhanced Trading Dashboard", layout="wide")
st.title("📊 Trading Dashboard with Buy/Sell Signals")

# Sidebar controls
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "1d"])
theme = st.sidebar.radio("Chart Theme", ["Light", "Dark"])

# Indicator toggles
show_ema200 = st.sidebar.checkbox("Show EMA 200", True)
show_ema50 = st.sidebar.checkbox("Show EMA 50", True)
show_sma20 = st.sidebar.checkbox("Show SMA 20", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_stoch = st.sidebar.checkbox("Show Stochastic", True)
show_volume = st.sidebar.checkbox("Show Volume", True)
show_signals = st.sidebar.checkbox("Show Buy/Sell Signals", True)

# Fetch data
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
period = "7d" if timeframe in ["1m", "5m", "15m", "1h"] else "1mo"
df = yf.download(tickers=symbol, interval=interval_map[timeframe], period=period)
if df.empty:
    st.error("No data available. Please try another symbol or timeframe.")
    st.stop()
df.reset_index(inplace=True)
df.rename(columns={"Date": "Datetime"}, inplace=True)
df.columns = [str(c).title() for c in df.columns]

# Indicators
if show_ema200:
    df["Ema200"] = ta.trend.ema_indicator(df["Close"], window=200)
if show_ema50:
    df["Ema50"] = ta.trend.ema_indicator(df["Close"], window=50)
if show_sma20:
    df["Sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
if show_macd:
    macd = ta.trend.MACD(df["Close"])
    df["Macd"] = macd.macd()
    df["Macd_Signal"] = macd.macd_signal()
    df["Macd_Hist"] = macd.macd_diff()
if show_stoch:
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

# Buy/Sell Signal Logic
buy_signals = []
sell_signals = []
if show_signals:
    for i in range(1, len(df)):
        try:
            buy_cond = (
                df["Close"].iloc[i] > df["Ema50"].iloc[i] and
                df["Macd"].iloc[i] > df["Macd_Signal"].iloc[i] and
                df["Stoch_K"].iloc[i-1] < df["Stoch_D"].iloc[i-1] and
                df["Stoch_K"].iloc[i] > df["Stoch_D"].iloc[i] and
                df["Stoch_K"].iloc[i] < 20
            )
            sell_cond = (
                df["Close"].iloc[i] < df["Ema50"].iloc[i] and
                df["Macd"].iloc[i] < df["Macd_Signal"].iloc[i] and
                df["Stoch_K"].iloc[i-1] > df["Stoch_D"].iloc[i-1] and
                df["Stoch_K"].iloc[i] < df["Stoch_D"].iloc[i] and
                df["Stoch_K"].iloc[i] > 80
            )
            if buy_cond:
                buy_signals.append((df["Datetime"].iloc[i], df["Low"].iloc[i]))
            elif sell_cond:
                sell_signals.append((df["Datetime"].iloc[i], df["High"].iloc[i]))
        except Exception:
            continue

# Chart
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
if show_ema200:
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Ema200"], line=dict(color="blue"), name="EMA 200"))
if show_ema50:
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Ema50"], line=dict(color="orange"), name="EMA 50"))
if show_sma20:
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Sma20"], line=dict(color="green"), name="SMA 20"))
if show_signals:
    for signal in buy_signals:
        fig.add_trace(go.Scatter(x=[signal[0]], y=[signal[1]], mode="markers+text", name="Buy", marker=dict(color="green", size=12), text=["BUY"], textposition="bottom center"))
    for signal in sell_signals:
        fig.add_trace(go.Scatter(x=[signal[0]], y=[signal[1]], mode="markers+text", name="Sell", marker=dict(color="red", size=12), text=["SELL"], textposition="top center"))
fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# MACD
if show_macd:
    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["Datetime"], y=df["Macd"], name="MACD", line=dict(color="cyan")))
    fig_macd.add_trace(go.Scatter(x=df["Datetime"], y=df["Macd_Signal"], name="Signal", line=dict(color="magenta")))
    fig_macd.add_trace(go.Bar(x=df["Datetime"], y=df["Macd_Hist"], name="Histogram", marker_color="gray"))
    fig_macd.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", height=300)
    st.plotly_chart(fig_macd, use_container_width=True)

# Stochastic
if show_stoch:
    st.subheader("Stochastic Oscillator")
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=df["Datetime"], y=df["Stoch_K"], name="%K", line=dict(color="blue")))
    fig_stoch.add_trace(go.Scatter(x=df["Datetime"], y=df["Stoch_D"], name="%D", line=dict(color="orange")))
    fig_stoch.add_hline(y=80, line=dict(dash="dot", color="red"))
    fig_stoch.add_hline(y=20, line=dict(dash="dot", color="green"))
    fig_stoch.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", height=300)
    st.plotly_chart(fig_stoch, use_container_width=True)

# Volume
if show_volume:
    st.subheader("Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], marker_color="gray", name="Volume"))
    fig_vol.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", height=250)
    st.plotly_chart(fig_vol, use_container_width=True)
