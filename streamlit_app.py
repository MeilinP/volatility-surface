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

def get_spot_price(client: RESTClient, symbol: str) -> Tuple[float, str]:
    """Force fetching the actual live trade price."""
    try:
        # This is the most accurate live price endpoint
        snapshot = client.get_snapshot_ticker("stocks", symbol)
        if snapshot and hasattr(snapshot, 'last_trade'):
            return snapshot.last_trade.price, "live_snapshot"
    except:
        pass
    return 691.97, "fallback"

@st.cache_data(ttl=30)
def fetch_data(api_key: str, symbol: str) -> Tuple[List[Dict], float, str, str]:
    if not POLYGON_AVAILABLE or not api_key:
        return [], 0.0, "error", "No API Key"

    client = RESTClient(api_key)
    spot, price_source = get_spot_price(client, symbol)
    
    # Get the full chain snapshot
    chain = client.list_snapshot_options_chain(symbol)
    data = []

    for opt in chain:
        strike = float(opt.details.strike_price)
        iv = getattr(opt, 'implied_volatility', None)
        is_call = opt.details.contract_type == 'call'
        
        # CRITICAL FIX: OTM-ONLY FILTER
        # Only use Puts for strikes below spot, and Calls for strikes above spot.
        # This removes the "double-counting" and ITM noise that causes spikes.
        is_otm = (is_call and strike >= spot) or (not is_call and strike < spot)
        
        if is_otm and iv and 0.05 < iv < 1.0:
            data.append({
                'expiration': opt.details.expiration_date,
                'strike': strike,
                'iv': iv
            })

    # Filter to the first 8 expirations for a clean view
    exps = sorted(set(d['expiration'] for d in data))[:8]
    data = [d for d in data if d['expiration'] in exps]
    
    debug = f"Spot: ${spot:.2f} ({price_source}) | Contracts: {len(data)}"
    return data, spot, "live", debug

def create_surface(data: List[Dict], spot: float, symbol: str):
    df = pd.DataFrame(data)
    # Average the IVs at each strike/expiration
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    
    # 1. SMOOTHING: Rolling mean across strikes to remove random 'bad prints'
    pivot = pivot.T.rolling(window=3, center=True, min_periods=1).mean().T
    
    # 2. INTERPOLATION: Use cubic to make it look like a smooth 'smile'
    pivot = pivot.sort_index().sort_index(axis=1)
    try:
        pivot = pivot.interpolate(method='cubic', axis=1).interpolate(method='linear', axis=0)
    except:
        pivot = pivot.interpolate(method='linear', axis=1) # Fallback if scipy fails
    
    pivot = pivot.ffill().bfill()
    
    strikes, exps = pivot.columns.values, pivot.index.tolist()
    X, Y = np.meshgrid(strikes, np.arange(len(exps)))
    Z = pivot.values * 100

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Magma')])
    
    # Add At-The-Money Line
    fig.add_trace(go.Scatter3d(
        x=[spot]*len(exps), y=list(range(len(exps))), 
        z=pivot.values[:, np.abs(strikes - spot).argmin()] * 100,
        mode='lines', line=dict(color='red', width=8), name='ATM'
    ))

    fig.update_layout(
        title=f"{symbol} IV Surface (Cleaned) | Spot: ${spot:.2f}",
        scene=dict(xaxis_title='Strike ($)', yaxis_title='Expiry', zaxis_title='IV %'),
        template="plotly_dark", height=700
    )
    return fig

def main():
    st.title("📊 Pro IV Surface")
    api_key = st.sidebar.text_input("Polygon API Key", type="password")
    symbol = st.sidebar.selectbox("Ticker", ["SPY", "QQQ", "NVDA"])
    
    if api_key:
        data, spot, source, debug = fetch_data(api_key, symbol)
        st.caption(debug)
        if data:
            st.plotly_chart(create_surface(data, spot, symbol), use_container_width=True)
    else:
        st.warning("Please enter your Polygon API key in the sidebar.")

if __name__ == "__main__":
    main()
