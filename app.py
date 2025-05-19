import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
from datetime import datetime
# ---------------------------------------------
# UI Config
# ---------------------------------------------
st.set_page_config(
    page_title="Advanced Trading Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.title("📊 Trading Dashboard")
    st.markdown("---")
    
    # Appearance Settings
    st.subheader("🎨 Appearance")
    theme = st.radio("Theme", ["Light", "Dark"], index=1)
    
    st.markdown("---")
    
    # Symbol & Timeframe selectors
    st.subheader("📈 Chart Settings")
    symbol = st.selectbox(
        "Select Symbol", 
        ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "META"]
    )
    
    timeframe = st.selectbox(
        "Select Timeframe", 
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"],
        index=4
    )
    
    st.markdown("---")
    
    # Indicator section with collapsible areas
    st.subheader("📊 Indicators")
    
    with st.expander("Price Indicators", expanded=True):
        show_sma = st.checkbox("Show SMA", value=True)
        if show_sma:
            sma_period = st.slider("SMA Period", 5, 500, 200, 5)
            
        show_ema = st.checkbox("Show EMA", value=True)
        if show_ema:
            ema_period = st.slider("EMA Period", 5, 200, 50, 5)
            
        # Removing Bollinger Bands as requested
        show_bollinger = False
    
    with st.expander("Momentum Indicators", expanded=True):
        show_macd = st.checkbox("Show MACD", value=True)
        if show_macd:
            macd_fast = st.slider("MACD Fast Period", 5, 100, 45, 1)
            macd_slow = st.slider("MACD Slow Period", 10, 200, 90, 1)
            macd_signal = st.slider("MACD Signal Period", 5, 50, 21, 1)
        
        show_rsi = st.checkbox("Show RSI", value=True)
        if show_rsi:
            rsi_period = st.slider("RSI Period", 5, 30, 14, 1)
            rsi_buy_threshold = st.slider("RSI Oversold", 10, 40, 30, 5)
            rsi_sell_threshold = st.slider("RSI Overbought", 60, 90, 70, 5)
    
    with st.expander("Chart Elements", expanded=True):
        show_volume = st.checkbox("Show Volume", value=True)
        show_fvg = st.checkbox("Show Fair Value Gaps", value=True)
        show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    
    st.markdown("---")
    
    # Signal settings
    st.subheader("🔔 Signal Settings")
    consensus_threshold_input = st.number_input(
        "Signal Consensus Threshold", 
        min_value=1, 
        max_value=4, 
        value=2, 
        step=1, 
        help="Number of indicators that must agree for a Buy/Sell signal."
    )
# Main content area
st.markdown(f"# {symbol} Trading Analysis")
st.markdown(f"<span style='color:gray; font-size:14px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>", unsafe_allow_html=True)
# ---------------------------------------------
# Data Fetching
# ---------------------------------------------
interval_map = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk"
}
# Adjust period based on timeframe
period_map = {
    "1m": "1d",
    "5m": "5d",
    "15m": "5d",
    "30m": "7d",
    "1h": "14d",
    "4h": "30d",
    "1d": "180d",
    "1wk": "2y"
}
period = period_map.get(timeframe, "30d")
# Show loading spinner while fetching data
with st.spinner(f'Fetching {symbol} data for {timeframe} timeframe...'):
    df = yf.download(tickers=symbol, interval=interval_map[timeframe], period=period)
# Check for valid data
if df.empty:
    st.error("❌ Failed to fetch data. Try a different symbol or timeframe.")
    st.stop()
# Reset the index to make the Datetime index a column
df.reset_index(inplace=True)
# Normalize all column names
normalized_columns = []
for col in df.columns:
    if isinstance(col, tuple):
        # For tuples like ('Close', 'BTC-USD') -> 'Close'
        name_parts = [part for part in col if part]  
        name = name_parts[0] if name_parts else str(col)  
        normalized_columns.append(name.strip().title())
    else: # Non-tuple column name
        col_name_str = str(col).strip().title()
        if col_name_str == 'Index' and 'Datetime' not in normalized_columns:
            normalized_columns.append('Datetime')  
        else:
            normalized_columns.append(col_name_str)
df.columns = normalized_columns
# Check for required columns
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']
missing_cols = [rc for rc in required_cols if rc not in df.columns]
if missing_cols:
    st.error(f"❌ Data fetched, but missing essential columns: {', '.join(missing_cols)}.\nAvailable columns: {', '.join(df.columns)}")
    st.stop()
# ---------------------------------------------
# Calculate Indicators
# ---------------------------------------------
# SMA
if show_sma:
    df["Sma"] = ta.trend.sma_indicator(df["Close"], window=sma_period)
# EMA
if show_ema:
    df["Ema"] = ta.trend.ema_indicator(df["Close"], window=ema_period)
# Removed Bollinger Bands as requested
# MACD
if show_macd:
    macd = ta.trend.MACD(df["Close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df["Macd"] = macd.macd()
    df["Macd_Signal"] = macd.macd_signal()
    df["Macd_Hist"] = macd.macd_diff()
# RSI
if show_rsi:
    df["Rsi"] = ta.momentum.RSIIndicator(df["Close"], window=rsi_period).rsi()
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
df['Bollinger_Buy_Signal'] = False
df['Bollinger_Sell_Signal'] = False
# SMA Signals
if show_sma and "Sma" in df.columns and "Close" in df.columns:
    df.loc[(df['Close'] > df['Sma']) & (df['Close'].shift(1) <= df['Sma'].shift(1)), 'Sma_Buy_Signal'] = True
    df.loc[(df['Close'] < df['Sma']) & (df['Close'].shift(1) >= df['Sma'].shift(1)), 'Sma_Sell_Signal'] = True
# MACD Signals
if show_macd and "Macd" in df.columns and "Macd_Signal" in df.columns:
    df.loc[(df['Macd'] > df['Macd_Signal']) & (df['Macd'].shift(1) <= df['Macd_Signal'].shift(1)), 'Macd_Buy_Signal'] = True
    df.loc[(df['Macd'] < df['Macd_Signal']) & (df['Macd'].shift(1) >= df['Macd_Signal'].shift(1)), 'Macd_Sell_Signal'] = True
# RSI Signals
if show_rsi and "Rsi" in df.columns:
    df.loc[(df['Rsi'] > rsi_buy_threshold) & (df['Rsi'].shift(1) <= rsi_buy_threshold), 'Rsi_Buy_Signal'] = True
    df.loc[(df['Rsi'] < rsi_sell_threshold) & (df['Rsi'].shift(1) >= rsi_sell_threshold), 'Rsi_Sell_Signal'] = True
# Removed Bollinger Bands Signals as requested
# Consolidated Signal Score
df['Signal_Score'] = 0
# Add all buy signals
if 'Sma_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Sma_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Sma_Sell_Signal'].astype(int)
if 'Macd_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Macd_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Macd_Sell_Signal'].astype(int)
if 'Rsi_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Rsi_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Rsi_Sell_Signal'].astype(int)
if 'Bollinger_Buy_Signal' in df.columns:
    df['Signal_Score'] += df['Bollinger_Buy_Signal'].astype(int)
    df['Signal_Score'] -= df['Bollinger_Sell_Signal'].astype(int)
# Determine Consolidated Signal
df['Consolidated_Signal'] = "Hold"
consensus_threshold = consensus_threshold_input
df.loc[df['Signal_Score'] >= consensus_threshold, 'Consolidated_Signal'] = "Buy"
df.loc[df['Signal_Score'] <= -consensus_threshold, 'Consolidated_Signal'] = "Sell"
# ---------------------------------------------
# Display Latest Signal
# ---------------------------------------------
if not df.empty and 'Consolidated_Signal' in df.columns and 'Close' in df.columns and 'Datetime' in df.columns:
    if len(df) > 0:
        latest_signal = df['Consolidated_Signal'].iloc[-1]
        latest_price = df['Close'].iloc[-1]
        latest_time = df['Datetime'].iloc[-1]
        
        signal_color = "green" if latest_signal == "Buy" else "red" if latest_signal == "Sell" else "#808080"
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("📢 Latest Signal")
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {'#0F1218' if theme == 'Dark' else '#F0F2F6'}; margin-bottom: 10px; border: 2px solid {signal_color};">
            <p style="font-size: 1.2em; margin-bottom: 2px; color: {'#FAFAFA' if theme == 'Dark' else '#262730'};">{symbol} @ {latest_price:.2f}</p>
            <p style="font-size: 1.5em; color: {signal_color}; font-weight: bold; margin-bottom: 2px;">{latest_signal.upper()}</p>
            <p style="font-size: 0.8em; color: {'#A0A0A0' if theme == 'Dark' else '#606060'};">{pd.to_datetime(latest_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📢 Latest Signal")
            st.info("No data available to determine signal.")
else:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📢 Latest Signal")
        st.info("Signal data not available (indicators might be off or insufficient data).")
# ---------------------------------------------
# Dashboard Metrics
# ---------------------------------------------
# Create a metrics row for key stats
if not df.empty:
    metric_cols = st.columns(4)
    
    # Current price and change
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
    price_change = ((current_price - prev_close) / prev_close) * 100
    
    # Volume stats
    current_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].mean()
    volume_change_pct = ((current_volume - avg_volume) / avg_volume) * 100
    
    # Volatility (simple measure using High-Low range)
    current_volatility = ((df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Low'].iloc[-1]) * 100
    avg_volatility = ((df['High'] - df['Low']) / df['Low']).mean() * 100
    
    # Display metrics
    metric_cols[0].metric(
        "Current Price", 
        f"${current_price:.2f}", 
        f"{price_change:.2f}%" if price_change else "0.00%"
    )
    
    # Calculate price range
    price_high = df['High'].max()
    price_low = df['Low'].min()
    metric_cols[1].metric(
        "Price Range", 
        f"${price_low:.2f} - ${price_high:.2f}",
        f"Range: {((price_high - price_low) / price_low) * 100:.2f}%"
    )
    
    metric_cols[2].metric(
        "Volume", 
        f"{current_volume:,.0f}", 
        f"{volume_change_pct:.2f}% vs Avg" if volume_change_pct else "0.00%"
    )
    
    metric_cols[3].metric(
        "Volatility", 
        f"{current_volatility:.2f}%", 
        f"{current_volatility - avg_volatility:.2f}% vs Avg"
    )
# ---------------------------------------------
# Plotly Chart
# ---------------------------------------------
# Create main chart with enhanced zooming capabilities
fig = go.Figure()
# Candlestick
fig.add_trace(go.Candlestick(
    x=df["Datetime"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price",
    hoverinfo="x+y+text",
    text=[f"O: {o:.2f}<br>H: {h:.2f}<br>L: {l:.2f}<br>C: {c:.2f}" for o, h, l, c in zip(df['Open'], df['High'], df['Low'], df['Close'])]
))
# SMA
if show_sma and "Sma" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Datetime"],
        y=df["Sma"],
        line=dict(color="blue", width=1.5),
        name=f"SMA ({sma_period})"
    ))
# EMA
if show_ema and "Ema" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Datetime"],
        y=df["Ema"],
        line=dict(color="purple", width=1.5),
        name=f"EMA ({ema_period})"
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
                fillcolor="rgba(255, 165, 0, 0.2)",
                opacity=0.3,
                line_width=0,
                layer="below"
            )
            
            # Add annotation text for larger gaps
            if abs(fvg_start - fvg_end) / min(fvg_start, fvg_end) > 0.02:  # Only annotate significant gaps
                fig.add_annotation(
                    x=a["Datetime"],
                    y=(fvg_start + fvg_end) / 2,
                    text="FVG",
                    showarrow=False,
                    font=dict(size=8, color="orange"),
                    opacity=0.7
                )
# Plot Buy/Sell Signals
if show_signals and 'Consolidated_Signal' in df.columns and 'Low' in df.columns and 'High' in df.columns:
    buy_signals_df = df[df['Consolidated_Signal'] == 'Buy']
    sell_signals_df = df[df['Consolidated_Signal'] == 'Sell']
    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals_df["Datetime"],
            y=buy_signals_df["Low"] * 0.985,
            mode="markers",
            marker=dict(
                symbol="triangle-up", 
                color="rgba(0, 255, 0, 0.9)", 
                size=12, 
                line=dict(width=1, color='DarkGreen')
            ),
            name="Buy Signal",
            hoverinfo="text",
            hovertext=[f"Buy Signal<br>{pd.to_datetime(row.Datetime).strftime('%Y-%m-%d %H:%M')}<br>Price: {row.Close:.2f}<br>Score: +{row.Signal_Score}" for _, row in buy_signals_df.iterrows()]
        ))
    
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals_df["Datetime"],
            y=sell_signals_df["High"] * 1.015,
            mode="markers",
            marker=dict(
                symbol="triangle-down", 
                color="rgba(255, 0, 0, 0.9)", 
                size=12, 
                line=dict(width=1, color='DarkRed')
            ),
            name="Sell Signal",
            hoverinfo="text",
            hovertext=[f"Sell Signal<br>{pd.to_datetime(row.Datetime).strftime('%Y-%m-%d %H:%M')}<br>Price: {row.Close:.2f}<br>Score: {row.Signal_Score}" for _, row in sell_signals_df.iterrows()]
        ))
# Enhanced Chart Layout with better zooming capabilities
fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark" if theme == "Dark" else "plotly_white",
    margin=dict(l=10, r=10, t=30, b=10),
    height=650,  # Increased height for better visibility
    hovermode="x unified",
    dragmode='zoom',  # Default to zoom mode for TradingView-like experience
    modebar=dict(
        orientation='v',
        activecolor='#FF4B4B',
        bgcolor='rgba(0,0,0,0.1)' if theme == "Dark" else 'rgba(255,255,255,0.1)'
    ),
    newshape=dict(line_color='#00bfff'),  # Color for drawing tools
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikedash='solid',
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128, 128, 128, 0.2)',
        showticklabels=True,
        rangeslider=dict(
            visible=False,  # Hide the default range slider
        ),
        # Enable extended zoom functionality
        range=[df["Datetime"].iloc[-min(100, len(df))], df["Datetime"].iloc[-1]],  # Default zoom to last 100 candles
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(150, 150, 150, 0.2)',
            activecolor='rgba(250, 70, 70, 0.8)',
            x=0,
            y=1.1,
        )
    ),
    yaxis=dict(
        showspikes=True,
        spikemode='across', 
        spikesnap='cursor',
        spikethickness=1.5,
        spikedash='solid',
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128, 128, 128, 0.2)',
        fixedrange=False,  # Allow y-axis zoom for better analysis
        autorange=True,
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.4)',
        zerolinewidth=1,
        side='right'  # Place price scale on right side like TradingView
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
# Enhanced zoom/pan buttons for TradingView-like experience
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='Default View',
                    method='relayout',
                    args=['xaxis.range', [df["Datetime"].iloc[-min(100, len(df))], df["Datetime"].iloc[-1]]]
                ),
                dict(
                    label='Zoom Out',
                    method='relayout',
                    args=['xaxis.range', [df["Datetime"].iloc[-min(200, len(df))], df["Datetime"].iloc[-1]]]
                ),
                dict(
                    label='Zoom In',
                    method='relayout',
                    args=['xaxis.range', [df["Datetime"].iloc[-min(50, len(df))], df["Datetime"].iloc[-1]]]
                ),
                dict(
                    label='Full Range',
                    method='relayout',
                    args=['xaxis.range', [df["Datetime"].iloc[0], df["Datetime"].iloc[-1]]]
                ),
            ],
            direction='left',
            pad={'r': 10, 't': 10},
            x=0.01,
            y=1.1,
            xanchor='left',
            yanchor='top',
            font=dict(size=10, color='white' if theme == "Dark" else 'black'),
            bgcolor='rgba(50, 50, 50, 0.7)' if theme == "Dark" else 'rgba(240, 240, 240, 0.7)',
            bordercolor='rgba(100, 100, 100, 0.5)',
            borderwidth=1
        ),
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='📈 Zoom',
                    method='relayout',
                    args=[{'dragmode': 'zoom'}]
                ),
                dict(
                    label='👆 Pan',
                    method='relayout',
                    args=[{'dragmode': 'pan'}]
                ),
            ],
            direction='left',
            pad={'r': 10, 't': 10},
            x=0.30,
            y=1.1,
            xanchor='left',
            yanchor='top',
            font=dict(size=10, color='white' if theme == "Dark" else 'black'),
            bgcolor='rgba(50, 50, 50, 0.7)' if theme == "Dark" else 'rgba(240, 240, 240, 0.7)',
            bordercolor='rgba(100, 100, 100, 0.5)',
            borderwidth=1
        )
    ]
)
# Display the main price chart
st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------
# MACD Panel
# ---------------------------------------------
if show_macd and all(col in df.columns for col in ["Macd", "Macd_Signal", "Macd_Hist"]):
    st.subheader(f"MACD ({macd_fast}, {macd_slow}, {macd_signal})")
    
    # Create MACD figure
    fig_macd = go.Figure()
    
    # Add MACD components
    fig_macd.add_trace(go.Scatter(
        x=df["Datetime"], 
        y=df["Macd"], 
        name="MACD Line",
        line=dict(color="blue", width=1.5)
    ))
    
    fig_macd.add_trace(go.Scatter(
        x=df["Datetime"], 
        y=df["Macd_Signal"], 
        name="Signal Line",
        line=dict(color="red", width=1.5)
    ))
    
    # Add histogram
    colors = ['green' if val >= 0 else 'red' for val in df["Macd_Hist"]]
    fig_macd.add_trace(go.Bar(
        x=df["Datetime"], 
        y=df["Macd_Hist"], 
        name="Histogram",
        marker_color=colors,
        opacity=0.7
    ))
    
    # Set the layout for MACD chart
    fig_macd.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.1),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid'
        ),
        xaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid',
            rangeslider=dict(visible=False),
        ),
    )
    
    st.plotly_chart(fig_macd, use_container_width=True)
# ---------------------------------------------
# RSI Panel
# ---------------------------------------------
if show_rsi and "Rsi" in df.columns:
    st.subheader("RSI (Relative Strength Index)")
    
    # Create RSI figure
    fig_rsi = go.Figure()
    
    # Add RSI line
    fig_rsi.add_trace(go.Scatter(
        x=df["Datetime"], 
        y=df["Rsi"], 
        name="RSI",
        line=dict(color="purple", width=1.5)
    ))
    
    # Add overbought/oversold reference lines
    fig_rsi.add_hline(y=rsi_sell_threshold, line=dict(color="red", dash="dot"), annotation_text=f"Overbought ({rsi_sell_threshold})")
    fig_rsi.add_hline(y=rsi_buy_threshold, line=dict(color="green", dash="dot"), annotation_text=f"Oversold ({rsi_buy_threshold})")
    
    # Center line
    fig_rsi.add_hline(y=50, line=dict(color="gray", dash="dash"), annotation_text="Neutral (50)")
    
    # RSI Layout
    fig_rsi.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(
            range=[0, 100],
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid'
        ),
        xaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid',
            rangeslider=dict(visible=False),
        ),
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
# ---------------------------------------------
# Volume Chart
# ---------------------------------------------
if show_volume and "Volume" in df.columns:
    st.subheader("Volume Analysis")
    
    # Create volume figure
    fig_vol = go.Figure()
    
    # Color volume bars based on price movement
    colors = ['green' if close >= open else 'red' for open, close in zip(df["Open"], df["Close"])]
    
    # Add volume bars
    fig_vol.add_trace(go.Bar(
        x=df["Datetime"],
        y=df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.8,
        hoverinfo="x+y",
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Volume: %{y:,.0f}",
        ])
    ))
    
    # Add moving average of volume
    ma_period = 20
    df['Volume_MA'] = df['Volume'].rolling(window=ma_period).mean()
    fig_vol.add_trace(go.Scatter(
        x=df["Datetime"],
        y=df["Volume_MA"],
        name=f"{ma_period}-period MA",
        line=dict(color='rgba(255, 255, 255, 0.5)' if theme == "Dark" else 'rgba(0, 0, 0, 0.5)', width=1.5),
        hoverinfo="x+y",
        hovertemplate="<br>".join([
            "Date: %{x}",
            f"{ma_period}-MA: %{{y:,.0f}}",
        ])
    ))
    
    # Format volume chart
    fig_vol.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.1),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid',
            showticklabels=True,
            side='right',
        ),
        xaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid',
            rangeslider=dict(visible=False),
        ),
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
# Add information about the dashboard
with st.expander("About This Dashboard"):
    st.markdown("""
    ### Trading Dashboard Features
    
    This advanced trading dashboard provides real-time market analysis with the following features:
    
    * **Price Chart** - Candlestick chart with TradingView-like zooming and navigation
    * **Technical Indicators** - SMA, EMA, MACD (45, 90, 21), RSI
    * **Signal Generation** - Identifies potential buy/sell opportunities
    * **Fair Value Gaps** - Highlights important price levels that may act as support/resistance
    * **Volume Analysis** - Visual representation of volume with moving average
    
    The dashboard uses data from Yahoo Finance and technical analysis calculations from the 'ta' library.
    
    **Note:** This dashboard is for informational purposes only and should not be considered financial advice.
    """)
# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray; font-size: 12px;'>Trading Dashboard | Data Source: Yahoo Finance | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
