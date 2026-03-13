"""
Implied Volatility Surface Visualizer
Author: Meilin Pan
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="IV Surface", page_icon="📊", layout="wide")

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

    rng = np.random.default_rng(42)
    data = []
    today = datetime.now()

    for days in [7, 14, 21, 30, 45, 60, 90, 120, 180]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        T = days / 365.0
        atm_vol = 0.13 + 0.08 * np.exp(-days / 40.0)

        for strike in np.linspace(spot * 0.75, spot * 1.25, 40):
            m = np.log(strike / spot)
            skew  = -0.20 * m
            smile =  0.90 * m ** 2
            noise = rng.normal(0, 0.0015)
            iv = float(np.clip(atm_vol + skew + smile + noise, 0.04, 0.95))
            data.append({
                'expiration': exp,
                'strike': round(strike, 2),
                'iv': iv,
                'type': 'call'
            })

    return data, spot, "demo"


def create_surface(data, spot, symbol, source):
    df = pd.DataFrame(data)
    pivot = (
        df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
        .sort_index().sort_index(axis=1)
        .interpolate(axis=1).interpolate(axis=0).bfill().ffill()
    )

    strikes = pivot.columns.values
    exps    = pivot.index.tolist()
    X, Y    = np.meshgrid(strikes, np.arange(len(exps)))
    Z       = pivot.values * 100

    hover = np.array([
        [f"Strike: ${strikes[j]:.2f}<br>Expiry: {exps[i]}<br>IV: {Z[i,j]:.1f}%"
         for j in range(len(strikes))]
        for i in range(len(exps))
    ])

    spot_idx = int(np.abs(strikes - spot).argmin())

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdYlBu_r', opacity=0.95,
        hoverinfo='text', text=hover,
        colorbar=dict(title='IV (%)', len=0.7, thickness=15,
                      tickfont=dict(color='white'), titlefont=dict(color='white'))
    ))
    fig.add_trace(go.Scatter3d(
        x=[spot] * len(exps), y=list(range(len(exps))), z=Z[:, spot_idx],
        mode='lines', line=dict(color='white', width=4),
        name=f'ATM ${spot:.0f}', hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text=f'{symbol} Implied Volatility Surface  |  Spot ${spot:.2f}',
            x=0.5, font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis=dict(title='Strike ($)', backgroundcolor='#0e1117',
                       gridcolor='#2a2a2a', color='white'),
            yaxis=dict(title='Expiration', backgroundcolor='#0e1117',
                       gridcolor='#2a2a2a', color='white',
                       ticktext=[e[5:] for e in exps],
                       tickvals=list(range(len(exps)))),
            zaxis=dict(title='IV (%)', backgroundcolor='#0e1117',
                       gridcolor='#2a2a2a', color='white',
                       range=[int(Z.min()) - 1, int(Z.max()) + 2]),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.0)),
            aspectratio=dict(x=1.5, y=1.2, z=0.8),
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=640,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(bgcolor='rgba(30,33,48,0.8)', font=dict(color='white'))
    )
    return fig


def create_skew(data, spot):
    df = pd.DataFrame(data)
    front_exp = sorted(df['expiration'].unique())[0]
    sk = df[df['expiration'] == front_exp].sort_values('strike')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sk['strike'], y=sk['iv'] * 100,
        mode='lines+markers',
        line=dict(color='#00d4ff', width=2.5),
        marker=dict(size=4),
        hovertemplate='$%{x:.0f}  IV: %{y:.1f}%<extra></extra>'
    ))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff6666", line_width=1.5,
                  annotation_text="ATM", annotation_font_color="#ff6666")
    fig.update_layout(
        title=dict(text=f'Volatility Smile  ({front_exp})', x=0.5,
                   font=dict(size=14, color='white')),
        xaxis=dict(title='Strike ($)', gridcolor='#2a2a2a', color='white'),
        yaxis=dict(title='IV (%)', gridcolor='#2a2a2a', color='white'),
        paper_bgcolor='#0e1117', plot_bgcolor='#161b27',
        font=dict(color='white'), height=380, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    return fig


def create_term(data, spot):
    df = pd.DataFrame(data)
    atm = df[(df['strike'] >= spot * 0.985) & (df['strike'] <= spot * 1.015)]
    if atm.empty:
        atm = df
    term = atm.groupby('expiration')['iv'].mean().sort_index() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(term))), y=term.values,
        mode='lines+markers',
        line=dict(color='#00ff99', width=2.5),
        marker=dict(size=9),
        hovertemplate='%{customdata}  ATM IV: %{y:.1f}%<extra></extra>',
        customdata=term.index.tolist()
    ))
    fig.update_layout(
        title=dict(text='ATM Term Structure', x=0.5,
                   font=dict(size=14, color='white')),
        xaxis=dict(title='Expiration', ticktext=[e[5:] for e in term.index],
                   tickvals=list(range(len(term))), gridcolor='#2a2a2a', color='white'),
        yaxis=dict(title='IV (%)', gridcolor='#2a2a2a', color='white'),
        paper_bgcolor='#0e1117', plot_bgcolor='#161b27',
        font=dict(color='white'), height=380, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Options volatility visualization | Meilin Pan</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption("Synthetic IV surface with realistic equity skew and term structure.")
        st.markdown("[GitHub](https://github.com/MeilinP) | [LinkedIn](https://linkedin.com/in/meilinp123)")

    data, spot, source = fetch_data(symbol)
    df = pd.DataFrame(data)
    atm_iv = df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]['iv'].mean() * 100
    put_skew = (
        df[(df['strike'] >= spot * 0.90) & (df['strike'] <= spot * 0.95)]['iv'].mean() -
        df[(df['strike'] >= spot * 0.99) & (df['strike'] <= spot * 1.01)]['iv'].mean()
    ) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", symbol)
    c2.metric("Spot", f"${spot:.2f}")
    c3.metric("ATM IV", f"{atm_iv:.1f}%")
    c4.metric("25Δ Put Skew", f"{put_skew:+.1f}%")

    st.markdown("---")
    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(create_skew(data, spot), use_container_width=True)
    right.plotly_chart(create_term(data, spot), use_container_width=True)


if __name__ == "__main__":
    main()
