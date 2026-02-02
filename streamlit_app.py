"""
Live Volatility Surface - Streamlit App
Real-time IV surface visualization with Polygon.io
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict

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
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def fetch_data(api_key: str, symbol: str):
    """Fetch spot price and IV data from Polygon."""
    if not POLYGON_AVAILABLE or not api_key:
        return generate_demo_data(symbol)

    try:
        client = RESTClient(api_key)
        
        # Real-time spot price using minute bars
        try:
            aggs = list(client.get_aggs(symbol, 1, "minute",
                datetime.now() - timedelta(hours=1), datetime.now(), limit=1, sort="desc"))
            spot = aggs[0].close if aggs else None
        except:
            aggs = list(client.get_aggs(symbol, 1, "day",
                datetime.now() - timedelta(days=5), datetime.now(), limit=1, sort="desc"))
            spot = aggs[0].close if aggs else None
        
        if not spot:
            return generate_demo_data(symbol)

        # Options chain
        chain = client.list_snapshot_options_chain(symbol)
        min_s, max_s = spot * 0.92, spot * 1.08
        data = []

        for opt in chain:
            strike = opt.details.strike_price
            iv = getattr(opt, 'implied_volatility', None)
            if iv and 0.01 < iv < 2.0 and min_s <= strike <= max_s:
                data.append({
                    'expiration': opt.details.expiration_date,
                    'strike': strike,
                    'iv': iv,
                    'type': opt.details.contract_type
                })

        if data:
            exps = sorted(set(d['expiration'] for d in data))[:8]
            data = [d for d in data if d['expiration'] in exps]
            return data, spot, "live"

        return generate_demo_data(symbol)

    except Exception as e:
        st.warning(f"API error: {str(e)[:80]}")
        return generate_demo_data(symbol)


def generate_demo_data(symbol: str):
    """Generate demo IV data."""
    spots = {'SPY': 585.0, 'QQQ': 510.0, 'AAPL': 245.0, 'MSFT': 420.0, 'NVDA': 135.0, 'TSLA': 395.0}
    spot = spots.get(symbol, 100.0)
    data = []
    today = datetime.now()

    for days in [7, 14, 21, 30, 45, 60, 90]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        T = days / 365.0
        for strike in np.linspace(spot * 0.90, spot * 1.10, 25):
            log_m = np.log(strike / spot)
            iv = 0.18 - 0.15 * log_m + 0.08 * log_m**2 + 0.025 * np.sqrt(T)
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': max(0.08, min(0.6, iv)), 'type': 'call'})

    return data, spot, "demo"


def create_surface(data: List[Dict], spot: float, symbol: str, source: str):
    """Create 3D IV surface."""
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    pivot = pivot.sort_index().sort_index(axis=1).interpolate(axis=1).interpolate(axis=0).bfill().ffill()
    
    strikes, exps = pivot.columns.values, pivot.index.tolist()
    X, Y = np.meshgrid(strikes, np.arange(len(exps)))
    Z = pivot.values * 100

    hover = np.array([[f"Strike: ${strikes[j]:.2f}<br>Expiry: {exps[i]}<br>IV: {Z[i,j]:.1f}%"
                       for j in range(len(strikes))] for i in range(len(exps))])

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Magma', opacity=0.95,
        hoverinfo='text', text=hover,
        colorbar=dict(title='IV (%)', len=0.75, thickness=15)))

    spot_idx = np.abs(strikes - spot).argmin()
    fig.add_trace(go.Scatter3d(x=[spot]*len(exps), y=list(range(len(exps))), z=Z[:, spot_idx],
        mode='lines', line=dict(color='red', width=5), name=f'ATM (${spot:.2f})', hoverinfo='skip'))

    label = '🟢 LIVE' if source == 'live' else '🟡 DEMO'
    fig.update_layout(
        title=dict(text=f'{symbol} IV Surface<br><sup>{label} | Spot: ${spot:.2f} | {datetime.now().strftime("%H:%M:%S")}</sup>', x=0.5, font=dict(size=20, color='white')),
        scene=dict(
            xaxis=dict(title='Strike ($)', backgroundcolor='#0e1117', gridcolor='#333', color='white'),
            yaxis=dict(title='Expiration', backgroundcolor='#0e1117', gridcolor='#333', color='white',
                       ticktext=[e[5:] for e in exps], tickvals=list(range(len(exps)))),
            zaxis=dict(title='IV (%)', backgroundcolor='#0e1117', gridcolor='#333', color='white'),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        paper_bgcolor='#0e1117', font=dict(color='white'), height=600, margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(30,33,48,0.8)')
    )
    return fig


def create_skew(data: List[Dict], spot: float):
    """Create skew chart."""
    df = pd.DataFrame(data)
    front = sorted(df['expiration'].unique())[0]
    skew = df[df['expiration'] == front].sort_values('strike')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew['strike'], y=skew['iv']*100, mode='lines+markers',
        line=dict(color='#00d4ff', width=3), marker=dict(size=5),
        hovertemplate='Strike: $%{x:.2f}<br>IV: %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff4444", line_width=2,
        annotation_text=f"Spot: ${spot:.2f}", annotation_position="top")

    fig.update_layout(
        title=dict(text=f'Front-Month Skew ({front})', x=0.5, font=dict(size=16, color='white')),
        xaxis_title='Strike ($)', yaxis_title='IV (%)',
        paper_bgcolor='#0e1117', plot_bgcolor='#1e2130', font=dict(color='white'),
        xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'), height=400, showlegend=False
    )
    return fig


def create_term(data: List[Dict], spot: float):
    """Create term structure chart."""
    df = pd.DataFrame(data)
    atm = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]
    if atm.empty:
        atm = df
    term = atm.groupby('expiration')['iv'].mean().sort_index() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(term))), y=term.values, mode='lines+markers',
        line=dict(color='#00ff88', width=3), marker=dict(size=10),
        hovertemplate='%{customdata}<br>IV: %{y:.1f}%<extra></extra>', customdata=term.index.tolist()))

    fig.update_layout(
        title=dict(text='ATM Term Structure', x=0.5, font=dict(size=16, color='white')),
        xaxis=dict(title='Expiration', ticktext=[e[5:] for e in term.index], tickvals=list(range(len(term))), gridcolor='#333'),
        yaxis=dict(title='IV (%)', gridcolor='#333'),
        paper_bgcolor='#0e1117', plot_bgcolor='#1e2130', font=dict(color='white'), height=400, showlegend=False
    )
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Live Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time options volatility visualization | Built by Meilin Pan</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        try:
            api_key = st.secrets["POLYGON_API_KEY"]
        except:
            api_key = os.environ.get('POLYGON_API_KEY', '')
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("**About**: Visualizes IV across strikes and expirations. Observe volatility smile, skew, and term structure patterns.")
        st.markdown("[GitHub](https://github.com/MeilinP) | [LinkedIn](https://linkedin.com/in/meilinp123)")

    data, spot, source = fetch_data(api_key, symbol)
    df = pd.DataFrame(data)
    atm_iv = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]['iv'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", symbol)
    c2.metric("Spot Price", f"${spot:.2f}")
    c3.metric("ATM IV", f"{atm_iv:.1f}%")
    c4.metric("Data Source", "🟢 Live" if source == "live" else "🟡 Demo")

    st.markdown("---")
    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(create_skew(data, spot), use_container_width=True)
    right.plotly_chart(create_term(data, spot), use_container_width=True)

    st.markdown(f"<div style='text-align:center;color:#666;font-size:0.8rem'>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(data)} contracts</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()