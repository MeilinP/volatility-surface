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

# Custom CSS for a clean dark interface
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

def get_spot_price(client: RESTClient, symbol: str) -> Tuple[float, str]:
    """Prioritizes real-time trade prices over daily aggregates."""
    # 1. Try Last Trade (Most accurate for current market)
    try:
        trade = client.get_last_trade(symbol)
        if trade and trade.price:
            return trade.price, "trade"
    except:
        pass
    
    # 2. Try Last Quote (Mid-price)
    try:
        quote = client.get_last_quote(symbol)
        if quote and quote.ask_price and quote.bid_price:
            return (quote.ask_price + quote.bid_price) / 2, "quote"
    except:
        pass
    
    # 3. Fallback to Aggregates (Daily)
    try:
        aggs = list(client.get_aggs(
            symbol, 1, "minute", 
            (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d'),
            limit=1, sort="desc"
        ))
        if aggs:
            return aggs[0].close, "minute_agg"
    except:
        pass
    
    return None, None

@st.cache_data(ttl=5) # Lower TTL for more 'live' updates
def fetch_data(api_key: str, symbol: str) -> Tuple[List[Dict], float, str, str]:
    if not POLYGON_AVAILABLE or not api_key:
        return generate_demo_data(symbol) + ("no_api",)

    try:
        client = RESTClient(api_key)
        spot, price_source = get_spot_price(client, symbol)
        
        if not spot:
            return generate_demo_data(symbol) + ("no_spot",)

        chain = client.list_snapshot_options_chain(symbol)
        
        # Narrowed range for cleaner visualization
        min_s, max_s = spot * 0.94, spot * 1.06
        data = []

        for opt in chain:
            strike = opt.details.strike_price
            if strike < min_s or strike > max_s:
                continue
            
            iv = getattr(opt, 'implied_volatility', None)
            # Filter out zero IV and extreme outliers that ruin scaling
            if iv is None or iv < 0.01 or iv > 1.0:
                continue

            data.append({
                'expiration': opt.details.expiration_date,
                'strike': float(strike),
                'iv': float(iv),
                'type': opt.details.contract_type
            })

        if len(data) < 15:
            return generate_demo_data(symbol) + ("insufficient_live_data",)

        # Take top 10 expirations for a meaningful surface
        exps = sorted(set(d['expiration'] for d in data))[:10]
        data = [d for d in data if d['expiration'] in exps]
        
        debug = f"spot={spot:.2f}({price_source}), valid={len(data)}"
        return data, spot, "live", debug

    except Exception as e:
        return generate_demo_data(symbol) + (f"error: {str(e)}",)

def generate_demo_data(symbol: str) -> Tuple[List[Dict], float, str]:
    spot = 695.0 if symbol == "SPY" else 100.0
    data = []
    today = datetime.now()
    for days in [7, 14, 21, 30, 45, 60]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        for strike in np.linspace(spot * 0.95, spot * 1.05, 20):
            iv = 0.15 + 0.1 * ((strike/spot - 1)**2)
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': iv, 'type': 'call'})
    return data, spot, "demo"

def create_surface(data: List[Dict], spot: float, symbol: str, source: str) -> go.Figure:
    df = pd.DataFrame(data)
    # Pivot and Interpolate to fill gaps in the option chain
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    
    strikes = pivot.columns.values
    exps = pivot.index.tolist()
    Z = pivot.values * 100
    X, Y = np.meshgrid(strikes, np.arange(len(exps)))

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='IV %')
    )])

    fig.update_layout(
        title=f"{symbol} Volatility Surface",
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis=dict(title='Expiration', ticktext=[e[5:] for e in exps], tickvals=list(range(len(exps)))),
            zaxis_title='IV (%)'
        ),
        template="plotly_dark",
        height=700
    )
    return fig

def create_skew(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    front = sorted(df['expiration'].unique())[0]
    skew = df[df['expiration'] == front].sort_values('strike')
    
    # Apply a rolling average to smooth the line
    skew['iv_smooth'] = skew['iv'].rolling(window=3, center=True).mean().fillna(skew['iv'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew['strike'], y=skew['iv_smooth']*100, mode='lines+markers', name="Front Month"))
    fig.add_vline(x=spot, line_dash="dash", line_color="red", annotation_text="Spot")
    fig.update_layout(title=f"Volatility Skew ({front})", template="plotly_dark", xaxis_title="Strike", yaxis_title="IV %")
    return fig

def create_term(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    # Filter for ATM (within 1% of spot)
    atm = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]
    term = atm.groupby('expiration')['iv'].mean() * 100

    fig = go.Figure(go.Scatter(x=term.index, y=term.values, mode='lines+markers', line=dict(color='#00ff88')))
    fig.update_layout(title="ATM Term Structure", template="plotly_dark", xaxis_title="Expiration", yaxis_title="IV %")
    return fig

def main():
    st.markdown('<h1 class="main-header">📊 Live Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Built by Meilin Pan | Real-time Market Analysis</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        api_key = st.text_input("Polygon API Key", type="password", value=os.environ.get('POLYGON_API_KEY', ''))
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    data, spot, source, debug = fetch_data(api_key, symbol)
    df = pd.DataFrame(data)
    
    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"${spot:.2f}")
    c2.metric("Contracts Found", len(data))
    c3.metric("Data Source", "LIVE" if source == "live" else "DEMO")
    st.caption(f"Debug Info: {debug}")

    # Charts
    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)
    
    col_left, col_right = st.columns(2)
    col_left.plotly_chart(create_skew(data, spot), use_container_width=True)
    col_right.plotly_chart(create_term(data, spot), use_container_width=True)

if __name__ == "__main__":
    main()