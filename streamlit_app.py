"""
Live Implied Volatility Surface
================================
Real-time IV surface visualization using demo data with realistic vol smile/skew/term structure.
Author: Meilin Pan
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict

st.set_page_config(page_title="Live IV Surface", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #fff; text-align: center; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


SPOT_PRICES = {
    'SPY': 600.0, 'QQQ': 520.0, 'AAPL': 230.0,
    'MSFT': 420.0, 'NVDA': 130.0, 'TSLA': 400.0
}


@st.cache_data(ttl=300)
def fetch_data(symbol: str):
    # Try yfinance for real spot price only; always use synthetic IV
    spot = None
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="5d")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
    except Exception:
        pass

    if not spot:
        spot = SPOT_PRICES.get(symbol, 100.0)

    data = []
    today = datetime.now()
    np.random.seed(42)

    for days in [7, 14, 21, 30, 45, 60, 90, 120, 180]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        T = days / 365.0

        # Term structure: front end elevated (VIX-like), back end flatter
        atm_vol = 0.14 + 0.07 * np.exp(-days / 45.0)

        for strike in np.linspace(spot * 0.75, spot * 1.25, 35):
            log_m = np.log(strike / spot)

            # Equity skew: OTM puts richly priced (fear premium)
            skew  = -0.30 * log_m
            # Smile: both wings elevated
            smile =  0.20 * log_m ** 2
            # Noise
            noise = np.random.normal(0, 0.002)

            iv = atm_vol + skew + smile + noise
            iv = float(np.clip(iv, 0.04, 0.90))

            data.append({
                'expiration': exp,
                'strike': round(strike, 2),
                'iv': iv,
                'type': 'call'
            })

    return data, spot, "demo"


def create_surface(data: List[Dict], spot: float, symbol: str, source: str):
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index='expiration', columns='strike', values='iv', aggfunc='mean'
    ).sort_index().sort_index(axis=1)
    pivot = pivot.interpolate(axis=1).interpolate(axis=0).bfill().ffill()

    strikes = pivot.columns.values
    exps    = pivot.index.tolist()
    X, Y    = np.meshgrid(strikes, np.arange(len(exps)))
    Z       = pivot.values * 100

    hover = np.array([
        [f"Strike: ${strikes[j]:.2f}<br>Expiry: {exps[i]}<br>IV: {Z[i,j]:.1f}%"
         for j in range(len(strikes))]
        for i in range(len(exps))
    ])

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Magma', opacity=0.95,
        hoverinfo='text', text=hover,
        colorbar=dict(title='IV (%)', len=0.75, thickness=15)
    ))

    # ATM line
    spot_idx = int(np.abs(strikes - spot).argmin())
    fig.add_trace(go.Scatter3d(
        x=[spot] * len(exps),
        y=list(range(len(exps))),
        z=Z[:, spot_idx],
        mode='lines',
        line=dict(color='red', width=5),
        name=f'ATM (${spot:.2f})',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text=f'{symbol} Implied Volatility Surface<br>'
                 f'<sup>🟡 DEMO | Spot: ${spot:.2f} | {datetime.now().strftime("%H:%M:%S")}</sup>',
            x=0.5, font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis=dict(title='Strike ($)', backgroundcolor='#0e1117', gridcolor='#333', color='white'),
            yaxis=dict(
                title='Expiration',
                backgroundcolor='#0e1117', gridcolor='#333', color='white',
                ticktext=[e[5:] for e in exps],
                tickvals=list(range(len(exps)))
            ),
            zaxis=dict(title='IV (%)', backgroundcolor='#0e1117', gridcolor='#333', color='white', range=[0, max(Z.max() + 2, 25)]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=620,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(30,33,48,0.8)')
    )
    return fig


def create_skew(data: List[Dict], spot: float):
    df = pd.DataFrame(data)
    front_exp = sorted(df['expiration'].unique())[0]
    skew_df = df[df['expiration'] == front_exp].sort_values('strike')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=skew_df['strike'],
        y=skew_df['iv'] * 100,
        mode='lines+markers',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=5),
        hovertemplate='Strike: $%{x:.2f}<br>IV: %{y:.1f}%<extra></extra>'
    ))
    fig.add_vline(
        x=spot, line_dash="dash", line_color="#ff4444", line_width=2,
        annotation_text=f"ATM ${spot:.2f}", annotation_position="top"
    )
    fig.update_layout(
        title=dict(text=f'Front-Month Skew ({front_exp})', x=0.5, font=dict(size=15, color='white')),
        xaxis_title='Strike ($)', yaxis_title='IV (%)',
        paper_bgcolor='#0e1117', plot_bgcolor='#1e2130',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        height=400, showlegend=False
    )
    return fig


def create_term(data: List[Dict], spot: float):
    df = pd.DataFrame(data)
    atm = df[(df['strike'] >= spot * 0.98) & (df['strike'] <= spot * 1.02)]
    if atm.empty:
        atm = df
    term = atm.groupby('expiration')['iv'].mean().sort_index() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(term))),
        y=term.values,
        mode='lines+markers',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=10),
        hovertemplate='%{customdata}<br>IV: %{y:.1f}%<extra></extra>',
        customdata=term.index.tolist()
    ))
    fig.update_layout(
        title=dict(text='ATM Term Structure', x=0.5, font=dict(size=15, color='white')),
        xaxis=dict(
            title='Expiration',
            ticktext=[e[5:] for e in term.index],
            tickvals=list(range(len(term))),
            gridcolor='#333'
        ),
        yaxis=dict(title='IV (%)', gridcolor='#333'),
        paper_bgcolor='#0e1117', plot_bgcolor='#1e2130',
        font=dict(color='white'), height=400, showlegend=False
    )
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Options volatility visualization | Built by Meilin Pan</p>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown(
            "Visualizes IV across strikes and expirations. "
            "Observe volatility smile, skew, and term structure."
        )
        st.markdown("[GitHub](https://github.com/MeilinP) | [LinkedIn](https://linkedin.com/in/meilinp123)")

    data, spot, source = fetch_data(symbol)
    df = pd.DataFrame(data)
    atm_iv = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]['iv'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", symbol)
    c2.metric("Spot Price", f"${spot:.2f}")
    c3.metric("ATM IV", f"{atm_iv:.1f}%")
    c4.metric("Data Source", "🟡 Demo")

    st.markdown("---")
    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(create_skew(data, spot), use_container_width=True)
    right.plotly_chart(create_term(data, spot), use_container_width=True)

    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:0.8rem'>"
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(data)} contracts"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
