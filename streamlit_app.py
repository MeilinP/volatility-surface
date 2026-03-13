import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Live IV Surface", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_data(symbol: str) -> Tuple[List[Dict], float, str, str]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)

    # Spot price
    spot = None
    try:
        hist = ticker.history(period="5d")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
    except Exception:
        pass
    if not spot:
        spot = {'SPY': 600.0, 'QQQ': 520.0, 'AAPL': 230.0,
                'MSFT': 420.0, 'NVDA': 130.0, 'TSLA': 400.0}.get(symbol, 100.0)

    min_s, max_s = spot * 0.92, spot * 1.08
    data = []

    try:
        today = datetime.now().date()
        expirations = [e for e in ticker.options
                       if (datetime.strptime(e, '%Y-%m-%d').date() - today).days >= 7][:8]
        for exp in expirations:
            chain = ticker.option_chain(exp)
            for df_opt, opt_type in [(chain.calls, 'call'), (chain.puts, 'put')]:
                for _, row in df_opt.iterrows():
                    strike = row['strike']
                    iv = row['impliedVolatility']
                    if not pd.notna(iv) or iv <= 0.01 or iv > 1.2:
                        continue
                    if strike < min_s or strike > max_s:
                        continue
                    data.append({
                        'expiration': exp,
                        'strike': float(strike),
                        'iv': float(iv),
                        'type': opt_type
                    })
    except Exception as e:
        return generate_demo_data(symbol) + (f"Error: {str(e)[:60]}",)

    if len(data) < 10:
        return generate_demo_data(symbol) + (f"Too few contracts ({len(data)})",)

    exps = sorted(set(d['expiration'] for d in data))[:8]
    data = [d for d in data if d['expiration'] in exps]

    return data, spot, "live", f"spot={spot:.2f}, contracts={len(data)}"


def generate_demo_data(symbol: str) -> Tuple[List[Dict], float, str]:
    spot = 695.0 if symbol == "SPY" else 100.0
    data = []
    today = datetime.now()
    for days in [7, 14, 21, 30, 45, 60, 90]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        for strike in np.linspace(spot * 0.95, spot * 1.05, 15):
            dist = (strike / spot - 1)
            iv = 0.12 + 0.4 * (dist ** 2) + (days / 365 * 0.05)
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': iv, 'type': 'call'})
    return data, spot, "demo"


def create_surface(data: List[Dict], spot: float, symbol: str) -> go.Figure:
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
    pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    smoothed = gaussian_filter(pivot.values, sigma=1.2)

    strikes = pivot.columns.values
    exps = pivot.index.tolist()
    Z = smoothed * 100

    hover_text = np.array(
        [[f"Strike: ${strikes[j]:.0f}<br>Expiry: {exps[i]}<br>IV: {Z[i,j]:.1f}%"
          for j in range(len(strikes))]
         for i in range(len(exps))],
        dtype=object
    )

    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=list(range(len(exps))),
        z=Z,
        colorscale='Magma',
        colorbar=dict(title=dict(text='IV %', font=dict(color='white')),
                      thickness=15, len=0.5,
                      tickfont=dict(color='white')),
        hoverinfo='text',
        text=hover_text,
        name=''
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
    skew['iv_smooth'] = skew['iv'].rolling(window=7, center=True, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew['strike'], y=skew['iv_smooth'] * 100,
                             mode='lines+markers', line=dict(color='#00d4ff', width=3)))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff4444",
                  annotation_text=f"Spot: {spot:.2f}")
    fig.update_layout(title=f"Front-Month Skew ({front})", template="plotly_dark",
                      height=400, xaxis_title="Strike", yaxis_title="IV %")
    return fig


def create_term(data: List[Dict], spot: float) -> go.Figure:
    df = pd.DataFrame(data)
    atm = df[(df['strike'] >= spot * 0.995) & (df['strike'] <= spot * 1.005)]
    if atm.empty:
        atm = df
    term = atm.groupby('expiration')['iv'].mean() * 100

    fig = go.Figure(go.Scatter(x=term.index, y=term.values, mode='lines+markers',
                               line=dict(color='#00ff88', width=3)))
    fig.update_layout(title="ATM Term Structure", template="plotly_dark",
                      height=400, xaxis_title="Expiration", yaxis_title="IV %")
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Live Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Built by Meilin Pan | Real-time Market Analysis</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox("Select Ticker", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    data, spot, source, debug = fetch_data(symbol)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot Price", f"${spot:.2f}")
    c2.metric("Contracts", len(data))
    c3.metric("Source", "🟢 LIVE" if source == "live" else "🟡 DEMO")
    c4.metric("Ticker", symbol)
    st.caption(f"**Debug Info:** {debug}")

    st.plotly_chart(create_surface(data, spot, symbol), use_container_width=True)

    col_left, col_right = st.columns(2)
    col_left.plotly_chart(create_skew(data, spot), use_container_width=True)
    col_right.plotly_chart(create_term(data, spot), use_container_width=True)


if __name__ == "__main__":
    main()
