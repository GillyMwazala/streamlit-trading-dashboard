import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# App configurations
st.set_page_config(
    page_title="Trading View AI Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to make it look more like TradingView but with better visibility
st.markdown("""
<style>
    /* General styling */
    .main {
        background-color: #131722;
        color: #ffffff; /* Brighter text for better readability */
    }
    
    /* Headers styling for better sectioning */
    h1, h2, h3 {
        color: #f8f9fa;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #2a2e39;
    }
    
    h1 {
        color: #29b6f6;
        font-size: 2.2rem;
    }
    
    h2 {
        color: #ffeb3b;
        font-size: 1.8rem;
    }
    
    h3 {
        color: #f06292;
        font-size: 1.4rem;
    }
    
    /* Button styling for better visibility */
    .stButton>button {
        background-color: #2962ff;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1e40af;
        box-shadow: 0 0 15px rgba(41, 98, 255, 0.5);
    }
    
    /* Metrics styling for dashboard numbers */
    .stMetric {
        background-color: #1e222d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2962ff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Indicator cards for trading signals */
    .css-1r6slb0 {  /* Metrics container */
        background-color: #1e222d;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2e39;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #ffffff;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2962ff;
        color: white;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e222d;
        border-right: 1px solid #2a2e39;
        padding: 1.5rem;
    }
    
    /* Dataframe styling for better data readability */
    .dataframe {
        background-color: #1e222d;
        font-family: 'Courier New', monospace;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #2962ff;
        color: white;
        padding: 8px;
        text-align: center;
    }
    
    .dataframe td {
        padding: 6px;
        border: 1px solid #2a2e39;
    }
    
    /* Warning/Status banner */
    .warning-banner {
        background-color: rgba(255, 87, 34, 0.2);
        border-left: 4px solid #ff5722;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    
    .success-banner {
        background-color: rgba(0, 200, 83, 0.2);
        border-left: 4px solid #00c853;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #2962ff;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 600;
    }
    
    div.stButton > button:hover {
        background-color: #1e53e5;
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background-color: #2a2e39;
        border-radius: 5px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* Info block styling */
    .stAlert {
        background-color: #2a2e39;
        border-radius: 4px;
        border-left-color: #2962ff;
        color: #ffffff !important;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    .css-10trblm {
        color: #ffffff !important;
    }
    
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Container styling */
    [data-testid="stVerticalBlock"] > div {
        border-radius: 5px;
        padding: 10px 0;
    }
    
    /* All text elements should be white for visibility */
    span, p, div, h1, h2, h3, h4, h5, h6, li, a {
        color: #ffffff !important;
    }
    
    /* Ensure selectbox text is visible */
    .st-bq {
        background-color: #2a2e39;
        color: white !important;
    }
    
    /* Fix code highlighting */
    code {
        color: #56d1d8 !important;
    }
    /* Additional explicit text color for Streamlit widgets */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        color: #ffffff !important;
        background-color: #2a2e39 !important; /* Ensure input backgrounds are also dark */
    }
    .stSelectbox div[data-baseweb="select"] > div, /* Selected value in selectbox */
    .stMultiSelect div[data-baseweb="select"] > div /* Selected values in multiselect */
     {
        color: #ffffff !important;
    }
    
    /* Dropdown menu items for selectbox/multiselect */
    div[data-baseweb="popover"] ul[role="listbox"] li {
        background-color: #1e222d !important;
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li:hover {
        background-color: #2962ff !important;
    }

    /* Ensure radio and checkbox labels are white */
    .stRadio label, .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Ensure st.info/success/warning/error text is explicitly white if not covered by .stAlert */
    .stAlert > div > div { /* Targeting the inner div that often holds the text */
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with professional styling
with st.sidebar:
    st.title("🚀 AI Trading View Pro")
    
    # Asset selection with more visual appeal
    st.header("💹 Market Selection")
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.selectbox(
            "Select Asset",
            options=["BTC-USD", "ETH-USD"],
            index=0
        )
    with col2:
        st.markdown("###") # Spacing
        if symbol == "BTC-USD":
            st.markdown("₿")
        else:
            st.markdown("Ξ")
    
    # Time frame selection with better organization
    st.header("⏱️ Time Settings")
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1h", "15m", "5m", "1m"],
            index=0
        )
    with col2:
        # Dynamic period options based on timeframe
        period_options = {
            "1d": ["1mo", "3mo", "6mo", "1y", "max"],
            "1h": ["1d", "3d", "7d", "1mo", "3mo"],
            "15m": ["1d", "3d", "7d", "1mo"],
            "5m": ["1d", "3d", "7d"],
            "1m": ["1d", "2d", "5d"]
        }
    
    # Chart type
    chart_type = st.radio(
        "Chart Type",
        options=["Candlestick", "OHLC", "Line"],
        index=0
    )
    
    # Period of data to fetch
    period_options = {
        "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        "1h": ["2d", "5d", "1mo", "3mo"],
        "15m": ["1d", "5d", "1mo"],
        "5m": ["1d", "5d", "1mo"],
        "1m": ["1d", "2d", "5d"]
    }
    
    period = st.selectbox(
        "Select Period",
        options=period_options.get(timeframe, ["1mo"]),
        index=0
    )
    
    # Volume display
    show_volume = st.checkbox("Show Volume", True)
    
    # Analysis Options
    st.header("AI Analysis")
    
    # Analysis type selection
    analysis_type = st.multiselect(
        "Select Analysis",
        options=[
            "Trend Analysis", 
            "Support/Resistance", 
            "Fair Value Gaps", 
            "Price Action", 
            "Market Sentiment",
            "Entry/Exit Points",
            "Volume Analysis",
            "Liquidity Zones"
        ],
        default=["Trend Analysis", "Support/Resistance", "Entry/Exit Points", "Price Action", "Market Sentiment"]
    )
    
    # Run analysis button
    run_analysis = st.button("Analyze Market", use_container_width=True)

# Helper Functions

@st.cache_data(ttl=60)  # Cache data for 1 minute only to get fresh updates
def fetch_market_data(ticker, interval, period):
    """Fetch market data with error handling"""
    try:
        st.info(f"Fetching latest market data for {ticker}...")
        data = yf.download(tickers=ticker, interval=interval, period=period)
        if data.empty:
            st.error(f"No data available for {ticker}")
            return None
        st.success(f"Successfully loaded latest {ticker} data")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def find_support_resistance(df, n_levels=5):
    """Find support and resistance levels using price extremes"""
    # Convert to numpy array for faster processing
    if 'Close' not in df.columns:
        return [], []
        
    prices = df['Close'].values
    
    # Find local minima and maxima
    supports = []
    resistances = []
    
    # Simple algorithm to identify significant levels
    for i in range(2, len(prices) - 2):
        # Support
        if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
           prices[i] < prices[i+1] and prices[i] < prices[i+2]:
            supports.append(prices[i])
            
        # Resistance
        if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
           prices[i] > prices[i+1] and prices[i] > prices[i+2]:
            resistances.append(prices[i])
    
    # Get most significant levels
    supports = sorted(supports)
    resistances = sorted(resistances, reverse=True)
    
    # Cluster nearby levels
    clustered_supports = []
    clustered_resistances = []
    
    if supports:
        # Simple clustering to find distinct levels
        threshold = (max(prices) - min(prices)) * 0.01  # 1% of price range
        
        current_level = supports[0]
        current_cluster = [current_level]
        
        for level in supports[1:]:
            if abs(level - current_level) < threshold:
                current_cluster.append(level)
            else:
                clustered_supports.append(sum(current_cluster) / len(current_cluster))
                current_level = level
                current_cluster = [current_level]
                
        clustered_supports.append(sum(current_cluster) / len(current_cluster))
    
    if resistances:
        # Do the same for resistance levels
        threshold = (max(prices) - min(prices)) * 0.01
        
        current_level = resistances[0]
        current_cluster = [current_level]
        
        for level in resistances[1:]:
            if abs(level - current_level) < threshold:
                current_cluster.append(level)
            else:
                clustered_resistances.append(sum(current_cluster) / len(current_cluster))
                current_level = level
                current_cluster = [current_level]
                
        clustered_resistances.append(sum(current_cluster) / len(current_cluster))
    
    # Return top 'n_levels' support and resistance levels
    return (clustered_supports[:n_levels], 
            clustered_resistances[:n_levels])

def find_fair_value_gaps(df):
    """Identify fair value gaps in price data"""
    if len(df) < 3:
        return []
    
    gaps = []
    
    # Look for gaps between candles
    for i in range(1, len(df) - 1):
        # Get individual scalar values to avoid Series comparison issues
        try:
            # Extract scalar values using .item() to avoid Series comparison
            low_next = df['Low'].iloc[i+1].item() if hasattr(df['Low'].iloc[i+1], 'item') else df['Low'].iloc[i+1]
            high_prev = df['High'].iloc[i-1].item() if hasattr(df['High'].iloc[i-1], 'item') else df['High'].iloc[i-1]
            low_prev = df['Low'].iloc[i-1].item() if hasattr(df['Low'].iloc[i-1], 'item') else df['Low'].iloc[i-1]
            high_next = df['High'].iloc[i+1].item() if hasattr(df['High'].iloc[i+1], 'item') else df['High'].iloc[i+1]
            
            # Bullish FVG
            if low_next > high_prev:
                # Gap between previous candle high and next candle low
                gaps.append({
                    'type': 'bullish',
                    'datetime': df.index[i],
                    'top': low_next,
                    'bottom': high_prev,
                    'mid': (low_next + high_prev) / 2
                })
                
            # Bearish FVG
            if high_next < low_prev:
                # Gap between previous candle low and next candle high
                gaps.append({
                    'type': 'bearish',
                    'datetime': df.index[i],
                    'top': low_prev,
                    'bottom': high_next,
                    'mid': (low_prev + high_next) / 2
                })
        except (ValueError, AttributeError, TypeError) as e:
            # Skip this iteration if there's an issue with the data
            continue
    
    return gaps

def get_ai_analysis(data, analysis_types, symbol, timeframe):
    """Get AI analysis from OpenAI API or fallback to local analysis if API fails"""
    try:
        # Format recent price data for the prompt
        recent_data = data.tail(10).copy()
        price_summary = f"Recent {symbol} prices:\n"
        
        # Safely format recent price data
        for idx, row in recent_data.iterrows():
            # Extract scalar values
            open_val = float(row['Open'])
            high_val = float(row['High'])
            low_val = float(row['Low'])
            close_val = float(row['Close'])
            vol_val = int(float(row['Volume']))
            
            # Format the timestamp
            if hasattr(idx, 'strftime'):
                timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = str(idx)
                
            price_summary += f"- {timestamp}: Open {open_val:.2f}, High {high_val:.2f}, Low {low_val:.2f}, Close {close_val:.2f}, Volume {vol_val}\n"
        
        # Current price info - extract scalar values
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        price_change = ((current_price / prev_price) - 1) * 100
        
        # Get high, low, volume as scalars
        high_val = float(data['High'].max())
        low_val = float(data['Low'].min())
        vol_val = int(float(data['Volume'].sum()))
        
        # Get SMA values for trend analysis
        sma9 = float(data['SMA_9'].iloc[-1]) if 'SMA_9' in data.columns else None
        sma20 = float(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns else None
        sma50 = float(data['SMA_50'].iloc[-1]) if 'SMA_50' in data.columns else None
        
        # Get MACD values
        macd = float(data['MACD'].iloc[-1]) if 'MACD' in data.columns else None
        macd_signal = float(data['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in data.columns else None
        macd_hist = float(data['MACD_Hist'].iloc[-1]) if 'MACD_Hist' in data.columns else None
        
        # Get RSI
        rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None
        
        # Current market stats
        market_stats = (
            f"Current Price: ${current_price:.2f}\n"
            f"24h Change: {price_change:.2f}%\n"
            f"24h High: ${high_val:.2f}\n"
            f"24h Low: ${low_val:.2f}\n"
            f"Volume: {vol_val:,}\n"
        )
        
        # Support/Resistance levels
        supports, resistances = find_support_resistance(data)
        
        # Convert support/resistance to scalar values
        scalar_supports = []
        for s in supports[:3]:
            if hasattr(s, 'item'):
                scalar_supports.append(float(s.item()))
            else:
                scalar_supports.append(float(s))
                
        scalar_resistances = []
        for r in resistances[:3]:
            if hasattr(r, 'item'):
                scalar_resistances.append(float(r.item()))
            else:
                scalar_resistances.append(float(r))
        
        # Format support/resistance info
        support_resistance_info = "Key Support Levels: " + ", ".join([f"${s:.2f}" for s in scalar_supports]) + "\n"
        support_resistance_info += "Key Resistance Levels: " + ", ".join([f"${r:.2f}" for r in scalar_resistances])
        
        # Fair Value Gaps
        fvgs = find_fair_value_gaps(data)
        fvg_info = "Recent Fair Value Gaps:\n"
        
        if fvgs:
            for fvg in fvgs[-3:]:
                try:
                    # Extract scalar values if needed
                    fvg_type = fvg['type']
                    
                    # Convert values to scalar
                    bottom = float(fvg['bottom']) if hasattr(fvg['bottom'], 'item') else float(fvg['bottom'])
                    top = float(fvg['top']) if hasattr(fvg['top'], 'item') else float(fvg['top'])
                    mid = float(fvg['mid']) if hasattr(fvg['mid'], 'item') else float(fvg['mid'])
                    
                    fvg_info += f"- {fvg_type.title()} FVG at ${mid:.2f} (range: ${bottom:.2f}-${top:.2f})\n"
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            fvg_info += "No significant fair value gaps detected.\n"
            
        # Check if we have API quota by making a small test request
        try:
            # Create a small test request to check if API is working
            test_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, this is a test request. Reply with just the word 'ok'."}],
                max_tokens=5
            )
            
            if hasattr(test_response, 'choices') and len(test_response.choices) > 0:
                # API is working, proceed with full analysis
                # Create the full analysis request prompt
                analysis_request = f"""
                As a professional market analyst, provide insights on {symbol} at {timeframe} timeframe.
                
                {market_stats}
                
                {support_resistance_info}
                
                {fvg_info}
                
                {price_summary}
                
                For each selected analysis type, provide:
                """
                
                # Add specific instructions for each analysis type
                if "Trend Analysis" in analysis_types:
                    analysis_request += """
                    - Trend Analysis: Current trend direction (bullish, bearish, or neutral), trend strength, key trend features, and potential trend continuation or reversal signals. Identify if we're in a larger trend or consolidation. Mention any chart patterns.
                    """
                
                if "Support/Resistance" in analysis_types:
                    analysis_request += """
                    - Support/Resistance: Identify key price levels where the asset has historically found support or resistance. Analyze the strength of these levels and their significance in the current market context. Focus on the most important levels and their likelihood of holding or breaking.
                    """
                
                if "Fair Value Gaps" in analysis_types:
                    analysis_request += """
                    - Fair Value Gaps: Identify unfilled price gaps (fair value gaps or imbalances) and explain their significance for the current market structure. Discuss how they might act as magnets for price and their potential as targets.
                    """
                
                if "Price Action" in analysis_types:
                    analysis_request += """
                    - Price Action: Examine recent candlestick patterns and price behavior. Identify signs of accumulation, distribution, or indecision. Look for candlestick formations that suggest market psychology.
                    """
                
                if "Market Sentiment" in analysis_types:
                    analysis_request += """
                    - Market Sentiment: Assess the overall market sentiment and institutional positioning. Describe potential institutional intent based on price action and volume. Consider possible manipulation patterns.
                    """
                    
                if "Entry/Exit Points" in analysis_types:
                    analysis_request += """
                    - Entry/Exit Points: Identify specific price levels that could serve as optimal entry and exit points based on current market structure.
                      For each entry point, specify:
                        - The type of entry (e.g., breakout, pullback to support, FVG fill).
                        - Key confirmation signals to watch for (e.g., candlestick pattern, volume spike, indicator crossover).
                        - Conditions that would invalidate this entry setup.
                      Include clear, recommended stop-loss levels (with rationale, e.g., below recent swing low, ATR-based) and at least two profit targets with risk-reward ratios.
                    """
                    
                if "Volume Analysis" in analysis_types:
                    analysis_request += """
                    - Volume Analysis: Examine recent volume patterns and their relationship to price movements. Look for volume spikes, volume divergences, and cumulative volume patterns that might indicate a potential trend reversal or continuation.
                    """
                    
                if "Liquidity Zones" in analysis_types:
                    analysis_request += """
                    - Liquidity Zones: Identify areas where large stop losses or pending orders may be clustered. These areas may be targets for price moves as the market hunts for liquidity. Explain how these zones might influence future price action.
                    """
                
                analysis_request += """
                
                Finally, provide a clear, actionable trading recommendation with these elements:
                1. Overall Market Bias: Bullish, Bearish, or Neutral, with a confidence level (Low, Medium, High).
                2. Primary Trade Setup:
                    - Suggested Entry Point(s): Specific price level(s) or zone.
                    - Entry Rationale: Detailed explanation linking to the analysis (e.g., "Entry on pullback to confirmed support at $X, coinciding with bullish divergence on RSI").
                    - Confirmation Signals: What specific chart events or indicator readings would confirm the entry?
                    - Stop Loss: Recommended price level and why (e.g., "SL at $Y, just below the 50-period SMA and recent swing low").
                    - Take Profit Targets: At least 2 price targets (e.g., TP1 at $Z1 targeting nearest resistance, TP2 at $Z2 for further extension).
                    - Risk-Reward Ratio: For each target.
                3. Alternative Scenarios: Briefly mention any secondary setups or what might invalidate the primary view.
                4. Timeframe: Expected duration for the primary setup to play out (e.g., intraday, 1-3 days, 1 week).
                
                Format your analysis for maximum readability with clear sections, bold text for key terms, and bullet points where appropriate. Be precise and avoid vague statements.
                """
                
                # Make the API call
                response = client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                    messages=[
                        {"role": "system", "content": "You are an expert financial market analyst specializing in technical analysis and trading strategies. Your insights are precise, actionable, and based on sound technical principles. Provide chart analysis like a professional trader would."},
                        {"role": "user", "content": analysis_request}
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                )
                
                return response.choices[0].message.content
            else:
                # API response looks wrong, use local analysis
                raise Exception("API response invalid")
                
        except Exception as api_error:
            # If API fails, provide local basic analysis
            analysis = f"""
            ## Technical Analysis for {symbol} - {timeframe}
            
            ### Market Overview
            
            **Current Price:** ${current_price:.2f}
            **24h Change:** {price_change:.2f}%
            **24h High:** ${high_val:.2f}
            **24h Low:** ${low_val:.2f}
            
            ### Technical Indicators
            """
            
            # Add technical indicator analysis based on calculated values
            if sma9 is not None and sma20 is not None and sma50 is not None:
                analysis += f"""
                **Moving Averages:**
                - SMA 9: ${sma9:.2f}
                - SMA 20: ${sma20:.2f}
                - SMA 50: ${sma50:.2f}
                
                **Trend Analysis:**
                """
                
                if sma9 > sma20 and sma20 > sma50:
                    analysis += "- Strong Uptrend - All moving averages aligned bullishly (SMA9 > SMA20 > SMA50)"
                elif sma9 < sma20 and sma20 < sma50:
                    analysis += "- Strong Downtrend - All moving averages aligned bearishly (SMA9 < SMA20 < SMA50)"
                elif sma9 > sma20 and sma20 < sma50:
                    analysis += "- Potential reversal - Short-term moving averages turning bullish"
                elif sma9 < sma20 and sma20 > sma50:
                    analysis += "- Potential reversal - Short-term moving averages turning bearish"
                else:
                    analysis += "- Consolidation/Sideways - No clear trend direction from moving averages"
            
            if macd is not None and macd_signal is not None:
                analysis += f"""
                
                **MACD Analysis:**
                - MACD Line: {macd:.4f}
                - Signal Line: {macd_signal:.4f}
                - Histogram: {macd_hist:.4f}
                
                **Momentum:**
                """
                
                if macd > macd_signal and macd_hist > 0:
                    analysis += "- Bullish momentum is increasing"
                elif macd > macd_signal and macd_hist < 0:
                    analysis += "- Bullish momentum is emerging"
                elif macd < macd_signal and macd_hist < 0:
                    analysis += "- Bearish momentum is increasing"
                elif macd < macd_signal and macd_hist > 0:
                    analysis += "- Bearish momentum is emerging"
            
            if rsi is not None:
                analysis += f"""
                
                **RSI Analysis:**
                - Current RSI: {rsi:.2f}
                
                **Overbought/Oversold:**
                """
                
                if rsi > 70:
                    analysis += "- Overbought conditions: Potential for a pullback or correction"
                elif rsi < 30:
                    analysis += "- Oversold conditions: Potential for a bounce or recovery"
                else:
                    analysis += f"- RSI in neutral territory ({rsi:.2f})"
            
            # Add support/resistance levels
            analysis += """
            
            ### Key Price Levels
            
            **Support Levels:**
            """
            
            for s in scalar_supports[:3]:
                distance = ((s / current_price) - 1) * 100
                analysis += f"- ${s:.2f} ({distance:.2f}% from current price)\n"
            
            analysis += """
            
            **Resistance Levels:**
            """
            
            for r in scalar_resistances[:3]:
                distance = ((r / current_price) - 1) * 100
                analysis += f"- ${r:.2f} ({distance:.2f}% from current price)\n"
            
            # Add trade recommendation based on technical indicators
            analysis += """
            
            ### Trade Recommendation
            """
            
            # Determine market direction
            direction = "Neutral"
            confidence = "Low"
            
            if sma9 is not None and sma20 is not None:
                if sma9 > sma20 and (macd is not None and macd > macd_signal) and (rsi is not None and rsi > 50 and rsi < 70):
                    direction = "Bullish"
                    confidence = "Medium"
                    if sma20 > sma50:
                        confidence = "High"
                elif sma9 < sma20 and (macd is not None and macd < macd_signal) and (rsi is not None and rsi < 50 and rsi > 30):
                    direction = "Bearish"
                    confidence = "Medium"
                    if sma20 < sma50:
                        confidence = "High"
            
            analysis += f"""
            **Market Direction:** {direction} (Confidence: {confidence})
            
            **Entry Strategy:**
            """
            
            if direction == "Bullish":
                # Calculate suitable entry, stop loss and targets
                entry = current_price * 0.99  # Slight discount to current price
                stop_loss = min(scalar_supports[0] if scalar_supports else current_price * 0.95, current_price * 0.95)
                tp1 = current_price * 1.05
                tp2 = current_price * 1.10
                
                analysis += f"""
                - **Entry Point:** ${entry:.2f} (slight pullback from current price)
                - **Stop Loss:** ${stop_loss:.2f} (below key support)
                - **Take Profit 1:** ${tp1:.2f} (Risk-to-Reward: 1:{((tp1-entry)/(entry-stop_loss)):.1f})
                - **Take Profit 2:** ${tp2:.2f} (Risk-to-Reward: 1:{((tp2-entry)/(entry-stop_loss)):.1f})
                - **Timeframe:** Short to medium-term (1-2 weeks)
                """
            elif direction == "Bearish":
                # Calculate suitable entry, stop loss and targets
                entry = current_price * 1.01  # Slight premium to current price on bounce
                stop_loss = max(scalar_resistances[0] if scalar_resistances else current_price * 1.05, current_price * 1.05)
                tp1 = current_price * 0.95
                tp2 = current_price * 0.90
                
                analysis += f"""
                - **Entry Point:** ${entry:.2f} (on small bounce)
                - **Stop Loss:** ${stop_loss:.2f} (above key resistance)
                - **Take Profit 1:** ${tp1:.2f} (Risk-to-Reward: 1:{((entry-tp1)/(stop_loss-entry)):.1f})
                - **Take Profit 2:** ${tp2:.2f} (Risk-to-Reward: 1:{((entry-tp2)/(stop_loss-entry)):.1f})
                - **Timeframe:** Short to medium-term (1-2 weeks)
                """
            else:
                analysis += """
                - Wait for clearer market direction before entering any trades
                - Focus on identifying key breakout or breakdown levels
                - Consider range-bound strategies between identified support and resistance
                """
            
            analysis += """
            
            ### Risk Warning
            
            This analysis is based on technical indicators only and should not be considered financial advice.
            Always manage your risk appropriately and never invest more than you can afford to lose.
            """
            
            return analysis
    
    except Exception as e:
        # Provide a very basic fallback if everything else fails
        return f"""
        ## Basic Market Overview for {symbol}
        
        **Current Price:** ${current_price:.2f}
        **24h Change:** {price_change:.2f}%
        
        ### Error Notice
        
        An error occurred while generating the detailed analysis: {str(e)}
        
        Please try again later or check different timeframes/symbols.
        """

# Main content
st.title(f"AI Trading View Analysis")

# Fetch market data
with st.spinner('Fetching market data...'):
    # Map timeframes to yfinance intervals
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
    
    df = fetch_market_data(
        ticker=symbol,
        interval=interval_map[timeframe],
        period=period
    )

# Check if data is available
if df is None or df.empty:
    st.error(f"No data available for {symbol} at {timeframe} timeframe. Please try another symbol or timeframe.")
    st.stop()

# Process dataframe and ensure 1-dimensional data
df = df.copy()
df_reset = df.reset_index()

# Handle different date column names
if 'Date' in df_reset.columns:
    df_reset.rename(columns={"Date": "Datetime"}, inplace=True)
elif 'Datetime' not in df_reset.columns:
    date_col = df_reset.columns[0]
    df_reset.rename(columns={date_col: "Datetime"}, inplace=True)

# Calculate some trading signals for the chart
with st.spinner('Calculating trading signals...'):
    # Add some basic indicators
    # 1. Simple Moving Averages
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, 0.001)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Generate Buy/Sell signals
    buy_signals = []
    sell_signals = []
    
    for i in range(2, len(df)):
        # Skip if not enough data
        if i < 50:
            continue
            
        # Try to extract scalar values to avoid Series issues
        try:
            # Buy signal: SMA 9 crosses above SMA 20 with positive MACD
            sma9_now = float(df['SMA_9'].iloc[i])
            sma9_prev = float(df['SMA_9'].iloc[i-1])
            sma20_now = float(df['SMA_20'].iloc[i])
            sma20_prev = float(df['SMA_20'].iloc[i-1])
            macd_now = float(df['MACD'].iloc[i])
            rsi_now = float(df['RSI'].iloc[i])
            
            # Buy condition: SMA9 crosses above SMA20 + RSI < 60 + MACD > 0
            buy_cond = (sma9_prev <= sma20_prev and sma9_now > sma20_now and 
                       rsi_now < 60 and macd_now > 0)
                       
            # Sell condition: RSI > 70 or MACD crosses below signal
            macd_signal_now = float(df['MACD_Signal'].iloc[i])
            macd_signal_prev = float(df['MACD_Signal'].iloc[i-1])
            macd_prev = float(df['MACD'].iloc[i-1])
            
            sell_cond = (rsi_now > 70 or 
                        (macd_prev >= macd_signal_prev and macd_now < macd_signal_now))
                        
            if buy_cond:
                # Record buy signal with datetime and price
                buy_signals.append((df.index[i], df['Low'].iloc[i] * 0.998))  # Slightly below the low for visibility
                
            if sell_cond:
                # Record sell signal with datetime and price
                sell_signals.append((df.index[i], df['High'].iloc[i] * 1.002))  # Slightly above the high for visibility
                
        except (ValueError, TypeError, IndexError) as e:
            continue

# Find support and resistance levels
supports, resistances = find_support_resistance(df)

# Find fair value gaps
fvgs = find_fair_value_gaps(df)

# Create the main chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.03, 
                   row_heights=[0.8, 0.2] if show_volume else [1, 0])

# Add the price chart
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df_reset["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
elif chart_type == "OHLC":
    fig.add_trace(go.Ohlc(
        x=df_reset["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
else:  # Line chart
    fig.add_trace(go.Scatter(
        x=df_reset["Datetime"],
        y=df["Close"],
        name="Price",
        line=dict(color='#2962ff', width=2)
    ), row=1, col=1)

# Add moving averages for trend indication
fig.add_trace(go.Scatter(
    x=df_reset["Datetime"],
    y=df["SMA_9"],
    line=dict(color="#29b6f6", width=1.5),
    name="SMA 9"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_reset["Datetime"],
    y=df["SMA_20"],
    line=dict(color="#ffeb3b", width=1.5),
    name="SMA 20"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_reset["Datetime"],
    y=df["SMA_50"],
    line=dict(color="#f06292", width=1.5),
    name="SMA 50"
), row=1, col=1)

# Add buy signals with improved visibility
if buy_signals:
    buy_x = []
    buy_y = []
    for signal in buy_signals:
        if isinstance(signal[0], pd.Timestamp):
            # Convert timestamp to datetime that matches df_reset
            dt = signal[0]
            if dt in df.index:
                buy_x.append(df_reset["Datetime"].iloc[df.index.get_loc(dt)])
                buy_y.append(signal[1])
        else:
            buy_x.append(signal[0])
            buy_y.append(signal[1])
            
    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x,
            y=buy_y,
            mode="markers+text",
            marker=dict(color="#00e676", size=15, symbol="triangle-up", line=dict(width=2, color="#ffffff")),
            text=["BUY"] * len(buy_x),
            textposition="bottom center",
            textfont=dict(color="#ffffff", size=10, family="Arial Black"),
            name="Buy Signal"
        ), row=1, col=1)

# Add sell signals with improved visibility
if sell_signals:
    sell_x = []
    sell_y = []
    for signal in sell_signals:
        if isinstance(signal[0], pd.Timestamp):
            # Convert timestamp to datetime that matches df_reset
            dt = signal[0]
            if dt in df.index:
                sell_x.append(df_reset["Datetime"].iloc[df.index.get_loc(dt)])
                sell_y.append(signal[1])
        else:
            sell_x.append(signal[0])
            sell_y.append(signal[1])
            
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x,
            y=sell_y,
            mode="markers+text",
            marker=dict(color="#ff1744", size=15, symbol="triangle-down", line=dict(width=2, color="#ffffff")),
            text=["SELL"] * len(sell_x),
            textposition="top center",
            textfont=dict(color="#ffffff", size=10, family="Arial Black"),
            name="Sell Signal"
        ), row=1, col=1)

# Add support levels with improved visibility
for level in supports:
    fig.add_shape(
        type="line",
        x0=df_reset["Datetime"].iloc[0],
        x1=df_reset["Datetime"].iloc[-1],
        y0=level,
        y1=level,
        line=dict(color="#26a69a", width=2, dash="dot"),  # Thicker line for better visibility
        row=1, col=1
    )

# Add resistance levels with improved visibility
for level in resistances:
    fig.add_shape(
        type="line",
        x0=df_reset["Datetime"].iloc[0],
        x1=df_reset["Datetime"].iloc[-1],
        y0=level,
        y1=level,
        line=dict(color="#ef5350", width=2, dash="dot"),  # Thicker line for better visibility
        row=1, col=1
    )

# Add fair value gaps
for fvg in fvgs:
    # Use rectangle shape for FVGs
    color = "rgba(38, 166, 154, 0.3)" if fvg['type'] == 'bullish' else "rgba(239, 83, 80, 0.3)"
    
    # Find the x-position (index) of the FVG in the data
    try:
        if isinstance(fvg['datetime'], pd.Timestamp):
            x_idx = df_reset[df_reset['Datetime'] == fvg['datetime']].index[0]
        else:
            # Handle case when datetime is already an index
            x_idx = fvg['datetime']
            
        # Calculate x positions
        if x_idx > 0 and x_idx < len(df_reset) - 1:
            x0 = df_reset["Datetime"].iloc[x_idx - 1]
            x1 = df_reset["Datetime"].iloc[x_idx + 1]
            
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=fvg['bottom'],
                y1=fvg['top'],
                line=dict(width=0),
                fillcolor=color,
                row=1, col=1
            )
    except (IndexError, KeyError):
        # Skip if we can't find the index
        continue

# Add volume chart if enabled
if show_volume:
    # Create color array for volume bars based on price direction
    colors = []
    for i in range(len(df)):
        try:
            # Extract scalar values to avoid Series comparison issues
            close_val = df['Close'].iloc[i].item() if hasattr(df['Close'].iloc[i], 'item') else df['Close'].iloc[i]
            open_val = df['Open'].iloc[i].item() if hasattr(df['Open'].iloc[i], 'item') else df['Open'].iloc[i]
            
            if close_val < open_val:
                colors.append('#ef5350')  # Red for bearish candles
            else:
                colors.append('#26a69a')  # Green for bullish candles
        except:
            # Default to neutral color if there's an issue
            colors.append('#808080')
    
    fig.add_trace(go.Bar(
        x=df_reset["Datetime"],
        y=df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

# Update chart layout with improved visibility
fig.update_layout(
    title=dict(
        text=f"{symbol} - {timeframe} Chart",
        font=dict(size=24, color="#ffffff")
    ),
    xaxis_title="",
    yaxis_title=dict(text="Price", font=dict(size=16, color="#ffffff")),
    xaxis_rangeslider_visible=False,
    height=650,  # Taller chart for better visibility
    template="plotly_dark",
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1,
        font=dict(size=12, color="#ffffff")
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(family="Arial, sans-serif", size=14, color="#ffffff"),
    paper_bgcolor="#131722",
    plot_bgcolor="#131722",
    hovermode="x unified"
)

# Make chart elements and grid lines more visible
fig.update_xaxes(
    showgrid=True,
    gridwidth=0.7,
    gridcolor="#2a2e39",  # Brighter grid color
    showline=True,
    linewidth=1.5,
    linecolor="#ffffff",  # White axis lines
    zeroline=False,
    title_font=dict(size=14, color="#ffffff"),
    tickfont=dict(size=12, color="#ffffff")
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=0.7,
    gridcolor="#2a2e39",  # Brighter grid color
    showline=True,
    linewidth=1.5,
    linecolor="#ffffff",  # White axis lines
    zeroline=False,
    title_font=dict(size=14, color="#ffffff"),
    tickfont=dict(size=12, color="#ffffff")
)

# Enable drawing tools and zooming by default
config = {
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ],
    'scrollZoom': True
}

# Display the chart
st.plotly_chart(fig, use_container_width=True, config=config)

# Market Stats
col1, col2, col3, col4 = st.columns(4)

# Extract scalar values to avoid Series comparison issues
try:
    current_price = df["Close"].iloc[-1].item() if hasattr(df["Close"].iloc[-1], 'item') else df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2].item() if hasattr(df["Close"].iloc[-2], 'item') else df["Close"].iloc[-2]
    
    price_change = ((current_price / prev_close) - 1) * 100
    price_color = "green" if price_change >= 0 else "red"
    
    # Calculate 24h stats
    if timeframe in ["1m", "5m", "15m"]:
        high_24h = df['High'].iloc[-24:].max()
        low_24h = df['Low'].iloc[-24:].min()
        volume_24h = df['Volume'].iloc[-24:].sum()
    else:
        high_24h = df['High'].max()
        low_24h = df['Low'].min()
        volume_24h = df['Volume'].sum()
    
    # Convert to scalar if needed
    high_24h = high_24h.item() if hasattr(high_24h, 'item') else high_24h
    low_24h = low_24h.item() if hasattr(low_24h, 'item') else low_24h
    volume_24h = volume_24h.item() if hasattr(volume_24h, 'item') else volume_24h
    
    col1.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{price_change:.2f}%"
    )
    
    col2.metric(
        "24h High",
        f"${high_24h:.2f}",
    )
    
    col3.metric(
        "24h Low",
        f"${low_24h:.2f}",
    )
    
    col4.metric(
        "Volume",
        f"{int(volume_24h):,}",
    )
    
except Exception as e:
    st.error(f"Error calculating market stats: {str(e)}")
    col1.metric("Current Price", "N/A")
    col2.metric("24h High", "N/A")
    col3.metric("24h Low", "N/A")
    col4.metric("Volume", "N/A")

# Market Analysis Section
st.header("Market Analysis")

# Generate AI Analysis when button is clicked
if run_analysis:
    with st.spinner("AI is analyzing the market..."):
        analysis = get_ai_analysis(df, analysis_type, symbol, timeframe)
        
    st.markdown(f"""
    <div style="background-color:#2a2e39; padding:20px; border-radius:5px; margin-top:10px;">
        <h3 style="color:#d1d4dc;">AI Market Analysis</h3>
        <div style="color:#d1d4dc; white-space: pre-line;">{analysis}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Click the 'Analyze Market' button for AI-powered analysis of trends, support/resistance levels, and fair value gaps.")

# Support & Resistance Levels
st.markdown("""
<h2 style="color:#ffffff; font-size:24px; font-weight:bold; margin-top:30px; margin-bottom:15px; text-align:center; background-color:#2a2e39; padding:15px; border-radius:5px;">
    🎯 Key Price Levels
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='color:#26a69a;'>Support Levels</h4>", unsafe_allow_html=True)
    
    # Process each support level safely
    for i, level in enumerate(supports[:5]):
        # Convert numpy arrays to Python scalars if needed
        if hasattr(level, 'item'):
            level_value = level.item()
        else:
            level_value = float(level)
            
        # Calculate distance safely
        try:
            # Get the current price from our earlier calculation, or compute it again if needed
            current_price_val = 0
            try:
                # Try to access the current price we calculated earlier
                current_price_val = current_price
            except NameError:
                # If not available, calculate it here
                if len(df) > 0:
                    curr_price = df["Close"].iloc[-1]
                    current_price_val = curr_price.item() if hasattr(curr_price, 'item') else float(curr_price)
                else:
                    current_price_val = 1  # Fallback
            
            # Now calculate the distance
            distance = ((level_value / current_price_val) - 1) * 100
            distance_str = f"{distance:.2f}% from current"
        except (TypeError, ValueError, ZeroDivisionError, NameError):
            distance_str = "N/A"
            
        # Display with proper formatting
        st.markdown(
            f"<div style='background:#1c2030; padding:10px; margin:5px 0; border-radius:5px; border-left:4px solid #26a69a;'>${level_value:.2f} <span style='color:#9ca3af; float:right;'>({distance_str})</span></div>", 
            unsafe_allow_html=True
        )

with col2:
    st.markdown("<h4 style='color:#ef5350;'>Resistance Levels</h4>", unsafe_allow_html=True)
    
    # Process each resistance level safely
    for i, level in enumerate(resistances[:5]):
        # Convert numpy arrays to Python scalars if needed
        if hasattr(level, 'item'):
            level_value = level.item()
        else:
            level_value = float(level)
            
        # Calculate distance safely
        try:
            # Get the current price from our earlier calculation, or compute it again if needed
            current_price_val = 0
            try:
                # Try to access the current price we calculated earlier
                current_price_val = current_price
            except NameError:
                # If not available, calculate it here
                if len(df) > 0:
                    curr_price = df["Close"].iloc[-1]
                    current_price_val = curr_price.item() if hasattr(curr_price, 'item') else float(curr_price)
                else:
                    current_price_val = 1  # Fallback
            
            # Now calculate the distance
            distance = ((level_value / current_price_val) - 1) * 100
            distance_str = f"{distance:.2f}% from current"
        except (TypeError, ValueError, ZeroDivisionError, NameError):
            distance_str = "N/A"
            
        # Display with proper formatting
        st.markdown(
            f"<div style='background:#1c2030; padding:10px; margin:5px 0; border-radius:5px; border-left:4px solid #ef5350;'>${level_value:.2f} <span style='color:#9ca3af; float:right;'>({distance_str})</span></div>", 
            unsafe_allow_html=True
        )

# Fair Value Gaps
st.markdown("""
<h2 style="color:#ffffff; font-size:24px; font-weight:bold; margin-top:30px; margin-bottom:15px; text-align:center; background-color:#2a2e39; padding:15px; border-radius:5px;">
    ⚡ Fair Value Gaps
</h2>
""", unsafe_allow_html=True)

if fvgs:
    for i, fvg in enumerate(fvgs[-5:]):  # Show the 5 most recent FVGs
        try:
            # Extract scalar values if needed
            fvg_type = fvg['type']
            color = "#26a69a" if fvg_type == "bullish" else "#ef5350"
            
            # Convert values to scalar if they're numpy arrays
            bottom = fvg['bottom'].item() if hasattr(fvg['bottom'], 'item') else float(fvg['bottom'])
            top = fvg['top'].item() if hasattr(fvg['top'], 'item') else float(fvg['top'])
            mid = fvg['mid'].item() if hasattr(fvg['mid'], 'item') else float(fvg['mid'])
            
            st.markdown(f"""
            <div style='background:#2a2e39; padding:15px; margin:10px 0; border-radius:5px; border-left:4px solid {color};'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:{color}; font-weight:600; font-size:18px;'>{fvg_type.title()} Fair Value Gap</span>
                    <span style='color:#ffffff; font-weight:600;'>Range: ${bottom:.2f} - ${top:.2f}</span>
                </div>
                <div style='margin-top:8px; color:#ffffff; font-size:16px;'>Mid-point: ${mid:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        except (KeyError, TypeError, ValueError) as e:
            # Skip problematic FVGs
            continue
else:
    st.info("No significant fair value gaps detected in the current timeframe.")

# Disclaimer
st.markdown("---")
st.caption("""
**Disclaimer**: This trading dashboard is for informational purposes only and does not constitute investment advice. 
The analysis and signals provided are based on technical indicators and AI algorithms that may not predict future market movements accurately. 
Always conduct your own research and consider consulting with a licensed financial advisor before making investment decisions.
""")
