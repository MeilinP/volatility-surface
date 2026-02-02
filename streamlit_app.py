import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

st.set_page_config(page_title="Live IV Surface", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

def get_spot_price(client: RESTClient, symbol: str) -> Tuple[float, str, str]:
    """Tries multiple methods to get the underlying price and returns (price, source, error_msg)."""
    error_msg = ""
    
    # 1. Try Last Trade
    try:
        trade = client.get_last_trade(symbol)
        if trade and trade.price:
            return trade.price, "trade", ""
    except Exception as e:
        error_msg += f"TradeErr: {str(e)[:30]}... "

    # 2. Try Last Quote
    try:
        quote = client.get_last_quote(symbol)
        if quote and quote.ask_price and quote.bid_price:
            return (quote.ask_price + quote.bid_price) / 2, "quote", ""
    except Exception as e:
        error_msg += f"QuoteErr: {str(e)[:30]}... "

    # 3. Try Daily Aggregate (Fallback for Free Tier)
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        aggs = list(client.get_aggs(symbol, 1, "day", start, end, limit=1, sort="desc"))
        if aggs:
            return aggs[0].close, "daily_agg", ""
    except Exception as e:
        error_msg += f"AggErr: {str(e)[:30]}"

    return None, None, error_msg

@st.cache_data(ttl=10)
def fetch_data(api_key: str, symbol: str) -> Tuple[List[Dict], float, str, str]:
    if not POLYGON_AVAILABLE:
        return generate_demo_data(symbol) + ("library_missing",)
    if not api_key:
        return generate_demo_data(symbol) + ("no_api_key",)

    try:
        client = RESTClient(api_key)
        spot, price_source, error_info = get_spot_price(client, symbol)
        
        # If we still don't have a spot price, we can't calculate moneyness
        if not spot:
            return generate_demo_data(symbol) + (f"Price error: {error_info}",)

        # Fetch the options chain
        chain = list(client.list_snapshot_options_chain(symbol))
        
        # Filtering parameters
        min_s, max_s = spot * 0.95, spot * 1.05
        data = []

        for opt in chain:
            strike = opt.details.strike_price
            # Filter by strike range
            if strike < min_s or strike > max_s:
                continue
            
            iv = getattr(opt, 'implied_volatility', None)
            # Clean out bad data points (0 IV or extreme spikes)
            if iv is None or iv <= 0.01 or iv > 1.2:
                continue

            data.append({
                'expiration': opt.details.expiration_date,
                'strike': float(strike),
                'iv': float(iv),
                'type': opt.details.contract_type
            })

        if len(data) < 10:
            return generate_demo_data(symbol) + (f"Too few contracts ({len(data)})",)

        # Sort and limit expirations for a cleaner 3D surface
        exps = sorted(set(d['expiration'] for d in data))[:8]
        data = [d for d in data if d['expiration'] in exps]
        
        debug = f"spot={spot:.2f}({price_source}), valid={len(data)}"
        return data, spot, "live", debug

    except Exception as e:
        return generate_demo_data(symbol) + (f"Global Error: {str(e)[:50]}",)

def generate_demo_data(symbol: str) -> Tuple[List[Dict], float, str]:
    spot = 695.0 if symbol == "SPY" else 100.0
    data = []
    today = datetime.now()
    for days in [7, 14, 21, 30, 45, 60, 90]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        for strike in np.linspace(spot * 0.95, spot * 1.05, 15):
            # Simulated smile: IV is higher OTM
            dist = (strike/spot - 1)
            iv = 0.12 + 0.4 * (dist**2) + (days/365 * 0.05)
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': iv, 'type': 'call'})
    return data, spot, "demo"

def create_surface(data: List[Dict], spot: float, symbol: str) -> go.Figure:
    df = pd.DataFrame(data)
    # Pivot and Interpolate
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    # Use linear interpolation to smooth the 'tears' in the surface
    pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    
    strikes = pivot.columns.values
    exps = pivot.index.tolist()
    Z = pivot.values * 100
    X, Y = np.meshgrid(strikes, np.arange(len(exps)))

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Magma',
        colorbar=dict(title='IV %', thickness=15, len=0.5)
    )])

    fig.update_layout(
        title=f"{symbol} Live Volatility Surface",
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis=dict(title='Expiry', ticktext=[e[5:] for e in exps], tickvals=list(range(len(exps)))),
            zaxis_title='IV (%)',
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50),
        height=600
    )
    return fig

def create_skew(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    front = sorted(df['expiration'].unique())[0]
    skew = df[df['expiration'] == front].sort_values('strike')
    
    # Simple smoothing to prevent zigzagging
    skew['iv_smooth'] = skew['iv'].rolling(window=3, center=True).mean().fillna(skew['iv'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew['strike'], y=skew['iv_smooth']*100, mode='lines+markers', line=dict(color='#00d4ff', width=3)))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff4444", annotation_text=f"Spot: {spot:.2f}")
    fig.update_layout(title=f"Front-Month Skew ({front})", template="plotly_dark", height=400, xaxis_title="Strike", yaxis_title="IV %")
    return fig

def create_term(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    # Filter for ATM (within 0.5% of spot)
    atm = df[(df['strike'] >= spot * 0.995) & (df['strike'] <= spot * 1.005)]
    if atm.empty: atm = df # Fallback if no exact ATM
    term = atm.groupby('expiration')['iv'].mean() * 100

    fig = go.Figure(go.Scatter(x=term.index, y=term.values, mode='lines+markers', line=dict(color='#00ff88', width=3)))
    fig.update_layout(title="ATM Term Structure", template="plotly_dark", height=400, xaxis_title="Expiration", yaxis_title="IV %")
    return fig

def main():
    st.markdown('<h1 class="main-header">📊 Live Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Built by Meilin Pan | Real-time Market Analysis</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox("Select Ticker", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        
        # User MUST enter API Key here if not in env vars
        api_key = st.text_input("Polygon API Key", type="password", help="Get a free key at polygon.io")
        if not api_key:
            api_key = os.environ.get('POLYGON_API_KEY', '')
            
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.info("Note: Free tier keys are limited to 5 calls per minute.")

    data, spot, source, debug = fetch_data(api_key, symbol)
    
    # Dashboard Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot Price", f"${spot:.2f}")
    c2.metric("Contracts", len(data))
    c3.metric("Source", "🟢 LIVE" if source == "live" else "🟡 DEMO")
    c4.metric("Ticker", symbol)
    
    st.caption(f"**Debug Info:** {debug}")

    if source == "demo" and "Price error" in debug:
        st.error(f"Failed to connect to Polygon. Showing Demo data. {debug}")

    # Layout: Surface on top, Skew and Term on bottom
    st.plotly_chart(create_surface(data, spot, symbol), use_container_width=True)
    
    col_left, col_right = st.columns(2)
    col_left.plotly_chart(create_skew(data, spot), use_container_width=True)
    col_right.plotly_chart(create_term(data, spot), use_container_width=True)

if __name__ == "__main__":
    main()
