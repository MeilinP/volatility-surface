"""
Implied Volatility Surface Visualizer
Author: Meilin Pan
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq


def _bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _calc_iv(price, S, K, T, r=0.045, option_type='call'):
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if price <= intrinsic:
        return np.nan
    try:
        return brentq(lambda s: _bs_price(S, K, T, r, s, option_type) - price,
                      0.001, 5.0, xtol=1e-5)
    except Exception:
        return np.nan

st.set_page_config(page_title="IV Surface", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #080c14; }

    .header-wrapper {
        background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0d1b2a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        text-align: center;
    }
    .header-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #7eb8f7, #a78bfa, #7eb8f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #546e8a;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    div[data-testid="metric-container"] {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: #546e8a !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0a1628;
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] * { color: #c0cfe0 !important; }
    .stSelectbox > div > div {
        background-color: #0d1b2a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    }

    hr { border-color: #1e3a5f !important; }
    .sidebar-badge {
        background: #0f2744;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #7eb8f7;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

SPOT_PRICES = {
    'SPY': 600.0, 'QQQ': 520.0, 'AAPL': 230.0,
    'MSFT': 420.0, 'NVDA': 130.0, 'TSLA': 400.0
}


def _synthetic_fallback(spot: float):
    # Calibrated for correct U-shaped smile with equity skew:
    # 7d:   ATM=18%, OTM put (75%)=50%, OTM call (125%)=20%  → skew=-0.434, smile=2.34
    # 180d: ATM=14%, OTM put (75%)=28%, OTM call (125%)=15%  → skew=-0.187, smile=1.04
    rng = np.random.default_rng(42)
    data = []
    today = datetime.now()
    for days in [7, 14, 21, 30, 45, 60, 90, 120, 180]:
        exp = (today + timedelta(days=days)).strftime('%Y-%m-%d')
        decay   = np.exp(-days / 30.0)
        atm_vol = 0.14 + 0.05 * decay
        skew    = -0.187 - 0.311 * decay
        smile   =  1.037 + 1.640 * decay
        for strike in np.linspace(spot * 0.75, spot * 1.25, 40):
            m = np.log(strike / spot)
            iv = float(np.clip(atm_vol + skew * m + smile * m ** 2 + rng.normal(0, 0.003), 0.04, 1.0))
            data.append({'expiration': exp, 'strike': round(strike, 2), 'iv': iv, 'type': 'call'})
    return data, "demo"


@st.cache_data(ttl=300)
def fetch_data(symbol: str):
    import yfinance as yf

    debug = []
    ticker = yf.Ticker(symbol)

    spot = None
    try:
        hist = ticker.history(period="5d")
        debug.append(f"history rows: {len(hist)}")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            debug.append(f"spot: {spot}")
    except Exception as e:
        debug.append(f"history error: {e}")
    if not spot:
        spot = SPOT_PRICES.get(symbol, 100.0)
        debug.append(f"spot fallback: {spot}")

    data = []
    try:
        expirations = ticker.options
        debug.append(f"expirations found: {len(expirations)} → {list(expirations[:3])}")
        if not expirations:
            raise ValueError("ticker.options is empty")

        today = datetime.now()
        for exp in expirations[:10]:
            exp_dt = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_dt - today).days
            if dte < 7:
                continue
            T = dte / 365.0
            chain = ticker.option_chain(exp)
            calls = chain.calls
            debug.append(f"  {exp} (dte={dte}): {len(calls)} calls")
            for _, row in calls.iterrows():
                strike = float(row['strike'])
                bid  = float(row['bid'])       if pd.notna(row['bid'])        else 0.0
                ask  = float(row['ask'])       if pd.notna(row['ask'])        else 0.0
                last = float(row['lastPrice']) if pd.notna(row['lastPrice'])  else 0.0

                if bid > 0 and ask > 0:
                    price = (bid + ask) / 2.0
                elif last > 0:
                    price = last
                else:
                    continue

                moneyness = strike / spot
                if not (0.80 <= moneyness <= 1.20 and price > 0.01):
                    continue

                iv = _calc_iv(price, spot, strike, T)
                if np.isnan(iv) or not (0.03 < iv < 2.0):
                    continue

                data.append({
                    'expiration': exp,
                    'strike': strike,
                    'iv': iv,
                    'type': 'call',
                    'dte': dte
                })

        debug.append(f"valid data points: {len(data)}")
        if len(data) > 20:
            return data, spot, "live", "\n".join(debug)
    except Exception as e:
        debug.append(f"options error: {e}")

    data, source = _synthetic_fallback(spot)
    return data, spot, source, "\n".join(debug)


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
        colorscale='Plasma', opacity=0.92,
        hoverinfo='text', text=hover,
        colorbar=dict(
            title=dict(text='IV (%)', font=dict(color='#8ba8c8', size=12)),
            len=0.65, thickness=12,
            tickfont=dict(color='#8ba8c8', size=11),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#1e3a5f', borderwidth=1
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=[spot] * len(exps), y=list(range(len(exps))), z=Z[:, spot_idx],
        mode='lines', line=dict(color='#7eb8f7', width=5),
        name=f'ATM ${spot:.0f}', hoverinfo='skip'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Strike ($)', backgroundcolor='#080c14',
                       gridcolor='#1e3a5f', color='#8ba8c8', linecolor='#1e3a5f'),
            yaxis=dict(title='', backgroundcolor='#080c14',
                       gridcolor='#1e3a5f', color='#8ba8c8',
                       ticktext=[e[5:] for e in exps],
                       tickvals=list(range(len(exps)))),
            zaxis=dict(title='IV (%)', backgroundcolor='#080c14',
                       gridcolor='#1e3a5f', color='#8ba8c8',
                       range=[int(Z.min()) - 1, int(Z.max()) + 2]),
            camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9)),
            aspectratio=dict(x=1.4, y=1.1, z=0.75),
            bgcolor='#080c14',
        ),
        paper_bgcolor='#080c14',
        font=dict(color='#8ba8c8', family='Inter'),
        height=580,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            bgcolor='rgba(13,27,42,0.9)',
            bordercolor='#1e3a5f', borderwidth=1,
            font=dict(color='#c0cfe0')
        ),
        hoverlabel=dict(bgcolor='#0d1b2a', bordercolor='#1e3a5f',
                        font=dict(color='white'))
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
        line=dict(color='#7eb8f7', width=2.5),
        marker=dict(size=5, color='#7eb8f7',
                    line=dict(color='#080c14', width=1)),
        fill='tozeroy', fillcolor='rgba(126,184,247,0.07)',
        hovertemplate='$%{x:.0f}  IV: %{y:.1f}%<extra></extra>'
    ))
    fig.add_vline(x=spot, line_dash="dash", line_color="#a78bfa", line_width=1.5,
                  annotation_text="ATM", annotation_font_color="#a78bfa",
                  annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f'Volatility Smile  ·  {front_exp}', x=0,
                   font=dict(size=13, color='#c0cfe0', family='Inter')),
        xaxis=dict(title='Strike ($)', gridcolor='#0d1b2a', color='#546e8a',
                   linecolor='#1e3a5f', tickformat='$,.0f'),
        yaxis=dict(title='IV (%)', gridcolor='#0d1b2a', color='#546e8a',
                   linecolor='#1e3a5f'),
        paper_bgcolor='#080c14', plot_bgcolor='#0a1220',
        font=dict(color='#8ba8c8', family='Inter'),
        height=340, showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        hoverlabel=dict(bgcolor='#0d1b2a', bordercolor='#1e3a5f',
                        font=dict(color='white'))
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
        line=dict(color='#a78bfa', width=2.5),
        marker=dict(size=8, color='#a78bfa', symbol='circle',
                    line=dict(color='#080c14', width=2)),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.07)',
        hovertemplate='%{customdata}  ATM IV: %{y:.1f}%<extra></extra>',
        customdata=term.index.tolist()
    ))
    fig.update_layout(
        title=dict(text='ATM Term Structure', x=0,
                   font=dict(size=13, color='#c0cfe0', family='Inter')),
        xaxis=dict(title='Expiration', ticktext=[e[5:] for e in term.index],
                   tickvals=list(range(len(term))),
                   gridcolor='#0d1b2a', color='#546e8a', linecolor='#1e3a5f'),
        yaxis=dict(title='IV (%)', gridcolor='#0d1b2a', color='#546e8a',
                   linecolor='#1e3a5f'),
        paper_bgcolor='#080c14', plot_bgcolor='#0a1220',
        font=dict(color='#8ba8c8', family='Inter'),
        height=340, showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        hoverlabel=dict(bgcolor='#0d1b2a', bordercolor='#1e3a5f',
                        font=dict(color='white'))
    )
    return fig


def main():
    st.markdown("""
    <div class="header-wrapper">
        <div class="header-title">Implied Volatility Surface</div>
        <div class="header-sub">Options analytics · Meilin Pan</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Settings")
        symbol = st.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"],
                              label_visibility="collapsed")
        st.markdown("")
        if st.button("↻  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-badge">
            Live spot via yfinance<br>Synthetic IV surface
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.caption("[GitHub](https://github.com/MeilinP) · [LinkedIn](https://linkedin.com/in/meilinp123)")

    data, spot, source, fetch_error = fetch_data(symbol)
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
    if source == "live":
        st.caption(f"✦ Live market data via yfinance · {len(data)} contracts")
    else:
        st.caption("⚠ Live data unavailable · showing synthetic surface")
        with st.expander("yfinance debug log"):
            st.code(fetch_error, language=None)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(create_surface(data, spot, symbol, source), use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(create_skew(data, spot), use_container_width=True)
    right.plotly_chart(create_term(data, spot), use_container_width=True)


if __name__ == "__main__":
    main()
