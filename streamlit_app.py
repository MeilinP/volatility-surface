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

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

def get_spot_price(client: RESTClient, symbol: str) -> Tuple[float, str]:
    """Retrieves the most recent live trade price."""
    try:
        # 1. Primary: Snapshot Ticker (Latest Trade)
        snapshot = client.get_snapshot_ticker("stocks", symbol)
        if snapshot and hasattr(snapshot, 'last_trade'):
            return snapshot.last_trade.price, "live_snapshot"
    except:
        pass
    
    try:
        # 2. Secondary: Last Trade
        trade = client.get_last_trade(symbol)
        if trade and trade.price:
            return trade.price, "last_trade"
    except:
        pass
    
    # 3. Final Fallback: Aggregates
    try:
        aggs = list(client.get_aggs(symbol, 1, "day", (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), limit=1, sort="desc"))
        if aggs: return aggs[0].close, "daily_close"
    except:
        pass
    return 100.0, "error_fallback"

@st.cache_data(ttl=30)
def fetch_data(api_key: str, symbol: str) -> Tuple[List[Dict], float, str, str]:
    if not POLYGON_AVAILABLE or not api_key:
        return generate_demo_data(symbol) + ("no_api",)

    try:
        client = RESTClient(api_key)
        spot, price_source = get_spot_price(client, symbol)
        
        chain = client.list_snapshot_options_chain(symbol)
        data = []

        for opt in chain:
            strike = opt.details.strike_price
            iv = getattr(opt, 'implied_volatility', None)
            is_call = opt.details.contract_type == 'call'
            
            # RULE: Filter for OTM only to remove noisy ITM data
            is_otm = (is_call and strike >= spot) or (not is_call and strike < spot)
            
            if is_otm and iv and 0.01 < iv < 1.2:
                data.append({
                    'expiration': opt.details.expiration_date,
                    'strike': float(strike),
                    'iv': float(iv),
                    'type': opt.details.contract_type
                })

        if len(data) < 20:
            return generate_demo_data(symbol) + ("insufficient_live_data",)

        # Limit to first 8 expirations for a clean surface
        exps = sorted(set(d['expiration'] for d in data))[:8]
        data = [d for d in data if d['expiration'] in exps]
        
        debug = f"Spot: ${spot:.2f} ({price_source}) | Contracts: {len(data)}"
        return data, spot, "live", debug

    except Exception as e:
        return generate_demo_data(symbol) + (f"Error: {str(e)[:50]}",)

def generate_demo_data(symbol: str) -> Tuple[List[Dict], float, str]:
    spot = 695.0 if symbol == "SPY" else 100.0
    data = []
    today = datetime.now()
    for days in [7, 14, 21, 30, 45, 60, 90]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        for strike in np.linspace(spot * 0.90, spot * 1.10, 20):
            dist = np.log(strike/spot)
            iv = 0.15 + 0.3 * (dist**2) + (days/365 * 0.04)
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': iv, 'type': 'call'})
    return data, spot, "demo"

def create_surface(data: List[Dict], spot: float, symbol: str, source: str) -> go.Figure:
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    
    # SMOOTHING: Rolling mean to remove random market noise spikes
    # Updated to avoid the deprecation warning
    pivot = pivot.T.rolling(window=3, center=True, min_periods=1).mean().T
    
    # INTERPOLATION: Cubic for the smooth professional look
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot = pivot.interpolate(method='cubic', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    
    strikes, exps = pivot.columns.values, pivot.index.tolist()
    X, Y = np.meshgrid(strikes, np.arange(len(exps)))
    Z = pivot.values * 100

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Magma', colorbar=dict(title='IV %', thickness=15))])
    
    # Add Spot Line
    spot_idx = np.abs(strikes - spot).argmin()
    fig.add_trace(go.Scatter3d(x=[spot]*len(exps), y=list(range(len(exps))), z=Z[:, spot_idx], mode='lines', line=dict(color='red', width=6), name=f'ATM (${spot:.2f})'))

    fig.update_layout(
        title=f"{symbol} IV Surface ({'LIVE' if source=='live' else 'DEMO'})",
        template="plotly_dark",
        scene=dict(
            xaxis_title='Strike ($)', 
            yaxis=dict(title='Expiry', ticktext=[e[5:] for e in exps], tickvals=list(range(len(exps)))),
            zaxis_title='IV %'
        ),
        height=600, margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig

def create_skew(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    front = sorted(df['expiration'].unique())[0]
    skew = df[df['expiration'] == front].sort_values('strike')
    
    # Simple smoothing for the 2D line
    skew['iv_smooth'] = skew['iv'].rolling(window=3, center=True, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew['strike'], y=skew['iv_smooth']*100, mode='lines+markers', line=dict(color='#00d4ff', width=3)))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff4444", annotation_text=f"Spot: {spot:.2f}")
    fig.update_layout(title=f"Front Skew ({front})", template="plotly_dark", height=400)
    return fig

def main():
    st.markdown('<h1 class="main-header">📊 Live IV Surface</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"])
        api_key = st.text_input("Polygon API Key", type="password") or os.environ.get('POLYGON_API_KEY', '')
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    data, spot, source, debug = fetch_data(api_key, symbol)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"${spot:.2f}")
    c2.metric("Status", "🟢 LIVE" if source == "live" else "🟡 DEMO")
    c3.metric("Symbol", symbol)
    st.caption(debug)

    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)
    
    l, r = st.columns(2)
    l.plotly_chart(create_skew(data, spot), use_container_width=True)
    # Re-using skew logic for a simple term structure view
    df = pd.DataFrame(data)
    term = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)].groupby('expiration')['iv'].mean()
    r.plotly_chart(go.Figure(go.Scatter(x=term.index, y=term.values*100, mode='lines+markers', line=dict(color='#00ff88'))).update_layout(title="ATM Term Structure", template="plotly_dark", height=400), use_container_width=True)

if __name__ == "__main__":
    main()
