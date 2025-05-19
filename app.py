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
show_signals = st.sidebar.checkbox("Show Buy/Sell Signals on Chart", value=True)
consensus_threshold_input = st.sidebar.number_input("Signal Consensus Threshold (1-3)", min_value=1, max_value=3, value=2, step=1, help="Number of indicators that must agree for a Buy/Sell signal.")

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
# Signal Generation
# ---------------------------------------------
# Initialize signal columns
df['Sma_Buy_Signal'] = False
df['Sma_Sell_Signal'] = False
df['Macd_Buy_Signal'] = False
df['Macd_Sell_Signal'] = False
df['Rsi_Buy_Signal'] = False
df['Rsi_Sell_Signal'] = False

# SMA Signals
if show_sma and "Sma200" in df.columns and "Close" in df.columns:
    df.loc[(df['Close'] > df['Sma200']) & (df['Close'].shift(1) <= df['Sma200'].shift(1)), 'Sma_Buy_Signal'] = True
    df.loc[(df['Close'] < df['Sma200']) & (df['Close'].shift(1) >= df['Sma200'].shift(1)), 'Sma_Sell_Signal'] = True

# MACD Signals
if show_macd and "Macd" in df.columns and "Macd_Signal" in df.columns:
    df.loc[(df['Macd'] > df['Macd_Signal']) & (df['Macd'].shift(1) <= df['Macd_Signal'].shift(1)), 'Macd_Buy_Signal'] = True
    df.loc[(df['Macd'] < df['Macd_Signal']) & (df['Macd'].shift(1) >= df['Macd_Signal'].shift(1)), 'Macd_Sell_Signal'] = True

# RSI Signals
if show_rsi and "Rsi" in df.columns:
    rsi_buy_threshold = 30
    rsi_sell_threshold = 70
    df.loc[(df['Rsi'] > rsi_buy_threshold) & (df['Rsi'].shift(1) <= rsi_buy_threshold), 'Rsi_Buy_Signal'] = True
    df.loc[(df['Rsi'] < rsi_sell_threshold) & (df['Rsi'].shift(1) >= rsi_sell_threshold), 'Rsi_Sell_Signal'] = True

# Consolidated Signal Score
df['Signal_Score'] = 0
# Ensure columns exist before trying to access them for scoring
if 'Sma_Buy_Signal' in df.columns: # These columns are always created now
    df['Signal_Score'] += df['Sma_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Sma_Sell_Signal'].astype(int)
if 'Macd_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Macd_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Macd_Sell_Signal'].astype(int)
if 'Rsi_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Rsi_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Rsi_Sell_Signal'].astype(int)

# Determine Consolidated Signal
df['Consolidated_Signal'] = "Hold"
consensus_threshold = consensus_threshold_input  # Use the user-defined threshold
df.loc[df['Signal_Score'] >= consensus_threshold, 'Consolidated_Signal'] = "Buy"
df.loc[df['Signal_Score'] <= -consensus_threshold, 'Consolidated_Signal'] = "Sell"

# ---------------------------------------------
# Display Latest Signal
# ---------------------------------------------
if not df.empty and 'Consolidated_Signal' in df.columns and 'Close' in df.columns and 'Datetime' in df.columns:
    if len(df) > 0: # Check if DataFrame has rows
        latest_signal = df['Consolidated_Signal'].iloc[-1]
        latest_price = df['Close'].iloc[-1]
        latest_time = df['Datetime'].iloc[-1]
        
        signal_color = "green" if latest_signal == "Buy" else "red" if latest_signal == "Sell" else "#808080" # Gray for Hold
        
        st.sidebar.subheader("Latest Signal:")
        st.sidebar.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #262730; margin-bottom: 10px; border: 1px solid {signal_color};">
        <p style="font-size: 1.0em; margin-bottom: 2px; color: #FAFAFA;">{symbol} at {latest_price:.2f}</p>
        <p style="font-size: 1.4em; color: {signal_color}; font-weight: bold; margin-bottom: 2px;">{latest_signal.upper()}</p>
        <p style="font-size: 0.8em; color: #808080;">{latest_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.subheader("Latest Signal:")
        st.sidebar.info("No data available to determine signal.")
else:
    st.sidebar.subheader("Latest Signal:")
    st.sidebar.info("Signal data not available (indicators might be off or insufficient data).")

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
    name="Price",
    hoverinfo="x+y+text", # Show x, y, and custom text
    text=[f"O: {o}<br>H: {h}<br>L: {l}<br>C: {c}" for o, h, l, c in zip(df['Open'], df['High'], df['Low'], df['Close'])] # Custom text for OHLC
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

# Plot Buy/Sell Signals
if show_signals and 'Consolidated_Signal' in df.columns and 'Low' in df.columns and 'High' in df.columns:
    buy_signals_df = df[df['Consolidated_Signal'] == 'Buy']
    sell_signals_df = df[df['Consolidated_Signal'] == 'Sell']

    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals_df["Datetime"],
            y=buy_signals_df["Low"] * 0.985, # Place slightly below the low
            mode="markers",
            marker=dict(symbol="triangle-up", color="rgba(0, 255, 0, 0.9)", size=10, line=dict(width=1, color='DarkGreen')),
            name="Buy Signal",
            hoverinfo="text",
            hovertext=[f"Buy Signal<br>{row.Datetime.strftime('%Y-%m-%d %H:%M')}<br>Price: {row.Close:.2f}" for index, row in buy_signals_df.iterrows()]
        ))
    
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals_df["Datetime"],
            y=sell_signals_df["High"] * 1.015, # Place slightly above the high
            mode="markers",
            marker=dict(symbol="triangle-down", color="rgba(255, 0, 0, 0.9)", size=10, line=dict(width=1, color='DarkRed')),
            name="Sell Signal",
            hoverinfo="text",
            hovertext=[f"Sell Signal<br>{row.Datetime.strftime('%Y-%m-%d %H:%M')}<br>Price: {row.Close:.2f}" for index, row in sell_signals_df.iterrows()]
        ))

# Chart Layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if theme == "Dark" else "plotly_white",
    margin=dict(l=0, r=0, t=30, b=0),
    height=600,
    hovermode="x unified",  # Unified hover information for all traces at a given x-value
    dragmode='pan', # Default to pan, zoom is available in modebar
    xaxis=dict(
        showspikes=True,  # Show spike line for x-axis
        spikemode='across', # Spike line goes across the plot area
        spikesnap='cursor', # Snap spike to cursor
        spikethickness=1,
        spikedash='dot'
    ),
    yaxis=dict(
        showspikes=True,  # Show spike line for y-axis
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikedash='dot'
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
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
