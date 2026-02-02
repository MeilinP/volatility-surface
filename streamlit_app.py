"""
Live Volatility Surface - Streamlit Web App
============================================

A deployable web application for visualizing implied volatility surfaces.
Deploy for free on Streamlit Cloud.

Author: Meilin Pan
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Try to import Polygon client
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Live IV Surface | Meilin Pan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Fetching
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_polygon_data(api_key: str, symbol: str) -> tuple:
    """Fetch live options data from Polygon.io"""
    
    if not POLYGON_AVAILABLE or not api_key or api_key == "DEMO":
        return None, None, "demo"
    
    try:
        client = RESTClient(api_key)
        
        # Get spot price
        aggs = list(client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=datetime.now() - timedelta(days=5),
            to=datetime.now(),
            limit=1,
            sort="desc"
        ))
        
        spot_price = aggs[0].close if aggs else None
        
        if not spot_price:
            return None, None, "no_spot"
        
        # Get options chain
        chain = client.list_snapshot_options_chain(underlying_asset=symbol)
        
        min_strike = spot_price * 0.92
        max_strike = spot_price * 1.08
        
        data = []
        for option in chain:
            try:
                strike = option.details.strike_price
                
                if not (min_strike <= strike <= max_strike):
                    continue
                
                iv = option.implied_volatility if hasattr(option, 'implied_volatility') else None
                
                if iv and 0.01 < iv < 2.0:
                    data.append({
                        'expiration': option.details.expiration_date,
                        'strike': strike,
                        'iv': iv,
                        'type': option.details.contract_type,
                    })
            except:
                continue
        
        if data:
            df = pd.DataFrame(data)
            expirations = sorted(df['expiration'].unique())[:8]
            data = [d for d in data if d['expiration'] in expirations]
        
        return data, spot_price, "live"
        
    except Exception as e:
        st.warning(f"Polygon API error: {str(e)[:100]}")
        return None, None, "error"


def generate_synthetic_data(symbol: str = "SPY") -> tuple:
    """Generate realistic synthetic IV data"""
    
    # Realistic spot prices
    spots = {'SPY': 585.0, 'QQQ': 510.0, 'AAPL': 245.0, 'MSFT': 420.0, 'NVDA': 135.0}
    spot_price = spots.get(symbol, 100.0)
    
    data = []
    base_vol = 0.18
    
    today = datetime.now()
    expirations = [(today + timedelta(days=d)).strftime('%Y-%m-%d') 
                   for d in [7, 14, 21, 30, 45, 60, 90]]
    
    strikes = np.linspace(spot_price * 0.90, spot_price * 1.10, 25)
    
    for exp_date in expirations:
        exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
        T = (exp_dt - today).days / 365.0
        
        for strike in strikes:
            moneyness = strike / spot_price
            log_m = np.log(moneyness)
            
            # Realistic IV model with smile and skew
            skew = -0.15 * log_m
            smile = 0.08 * log_m ** 2
            term = 0.025 * np.sqrt(max(T, 0.01))
            
            iv = base_vol + skew + smile + term
            iv = max(0.08, min(0.60, iv))
            
            data.append({
                'expiration': exp_date,
                'strike': round(strike, 2),
                'iv': iv,
                'type': 'call' if strike >= spot_price else 'put'
            })
    
    return data, spot_price, "synthetic"


# =============================================================================
# Visualization
# =============================================================================

def create_3d_surface(data: List[Dict], spot_price: float, symbol: str, data_source: str) -> go.Figure:
    """Create interactive 3D volatility surface with Plotly"""
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='expiration',
        columns='strike',
        values='iv',
        aggfunc='mean'
    ).sort_index().sort_index(axis=1)
    
    # Interpolate
    pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0)
    pivot = pivot.bfill().ffill()
    
    # Create meshgrid
    strikes = pivot.columns.values
    expirations = pivot.index.tolist()
    X, Y = np.meshgrid(strikes, np.arange(len(expirations)))
    Z = pivot.values * 100  # Convert to percentage
    
    # Create pre-built hover text for each point
    hover_text = []
    for i, exp in enumerate(expirations):
        row = []
        for j, strike in enumerate(strikes):
            iv_val = Z[i, j]
            row.append(f"<b>Strike:</b> ${strike:.2f}<br><b>Expiry:</b> {exp}<br><b>IV:</b> {iv_val:.1f}%")
        hover_text.append(row)
    hover_text = np.array(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Magma',
        opacity=0.95,
        name='IV Surface',
        colorbar=dict(
            title=dict(text='IV (%)', side='right'),
            len=0.75,
            thickness=15
        ),
        hoverinfo='text',
        text=hover_text
    ))
    
    # Add spot price line on surface
    spot_idx = np.abs(strikes - spot_price).argmin()
    fig.add_trace(go.Scatter3d(
        x=[spot_price] * len(expirations),
        y=list(range(len(expirations))),
        z=Z[:, spot_idx],
        mode='lines',
        line=dict(color='red', width=5),
        name=f'ATM (${spot_price:.2f})',
        hoverinfo='skip'
    ))
    
    # Determine data source label
    source_labels = {
        'live': '🟢 LIVE DATA',
        'synthetic': '🟡 DEMO DATA',
        'demo': '🟡 DEMO DATA'
    }
    source_label = source_labels.get(data_source, '🟡 DEMO')
    
    fig.update_layout(
        title=dict(
            text=f'{symbol} Implied Volatility Surface<br><sup>{source_label} | Spot: ${spot_price:.2f} | {datetime.now().strftime("%H:%M:%S")}</sup>',
            x=0.5,
            font=dict(size=20, color='white')
        ),
        scene=dict(
            xaxis_title='Strike Price ($)',
            yaxis_title='Expiration',
            zaxis_title='Implied Volatility (%)',
            xaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='#333',
                color='white'
            ),
            yaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='#333',
                color='white',
                ticktext=[e[5:] for e in expirations],  # MM-DD format
                tickvals=list(range(len(expirations)))
            ),
            zaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='#333',
                color='white'
            ),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            aspectratio=dict(x=1.2, y=1, z=0.6)
        ),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=80, b=0),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(30,33,48,0.8)'
        )
    )
    
    return fig


def create_skew_chart(data: List[Dict], spot_price: float, symbol: str) -> go.Figure:
    """Create 2D volatility skew chart for front-month"""
    
    df = pd.DataFrame(data)
    front_exp = sorted(df['expiration'].unique())[0]
    skew_data = df[df['expiration'] == front_exp].sort_values('strike')
    
    fig = go.Figure()
    
    # Add IV line
    fig.add_trace(go.Scatter(
        x=skew_data['strike'],
        y=skew_data['iv'] * 100,
        mode='lines+markers',
        name='Implied Volatility',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=6),
        hovertemplate='Strike: $%{x:.2f}<br>IV: %{y:.1f}%<extra></extra>'
    ))
    
    # Add spot line
    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="#ff4444",
        line_width=2,
        annotation_text=f"Spot: ${spot_price:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(
            text=f'Front-Month Volatility Skew ({front_exp})',
            x=0.5,
            font=dict(size=16, color='white')
        ),
        xaxis_title='Strike Price ($)',
        yaxis_title='Implied Volatility (%)',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e2130',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333', color='white'),
        yaxis=dict(gridcolor='#333', color='white'),
        height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def create_term_structure(data: List[Dict], spot_price: float) -> go.Figure:
    """Create ATM term structure chart"""
    
    df = pd.DataFrame(data)
    
    # Get ATM options (within 1% of spot)
    atm_df = df[(df['strike'] >= spot_price * 0.99) & (df['strike'] <= spot_price * 1.01)]
    
    if atm_df.empty:
        atm_df = df[df['strike'] == df['strike'].iloc[(df['strike'] - spot_price).abs().argmin()]]
    
    term_structure = atm_df.groupby('expiration')['iv'].mean().sort_index() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(term_structure))),
        y=term_structure.values,
        mode='lines+markers',
        name='ATM IV',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=10),
        hovertemplate='%{customdata}<br>IV: %{y:.1f}%<extra></extra>',
        customdata=term_structure.index.tolist()
    ))
    
    fig.update_layout(
        title=dict(
            text='ATM Volatility Term Structure',
            x=0.5,
            font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title='Expiration',
            ticktext=[e[5:] for e in term_structure.index],
            tickvals=list(range(len(term_structure))),
            gridcolor='#333',
            color='white'
        ),
        yaxis=dict(
            title='Implied Volatility (%)',
            gridcolor='#333',
            color='white'
        ),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e2130',
        font=dict(color='white'),
        height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Live Implied Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time options volatility visualization | Built by Meilin Pan</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        symbol = st.selectbox(
            "Select Symbol",
            ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"],
            index=0
        )
        
        # API Key input (optional)
        api_key = st.text_input(
            "Polygon API Key (optional)",
            value=os.environ.get('POLYGON_API_KEY', ''),
            type="password",
            help="Enter your Polygon.io API key for live data. Leave blank for demo mode."
        )
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        This tool visualizes the **implied volatility surface** - 
        showing how IV varies across:
        - **Strike prices** (moneyness)
        - **Expiration dates** (term structure)
        
        Key patterns to observe:
        - **Volatility Smile**: Higher IV for OTM options
        - **Skew**: OTM puts > OTM calls (crash protection)
        - **Term Structure**: IV changes with time
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[GitHub](https://github.com/meilinp) | [LinkedIn](https://www.linkedin.com/in/meilinp123/)")
    
    # Fetch data
    with st.spinner("Fetching options data..."):
        data, spot_price, data_source = fetch_polygon_data(api_key, symbol)
        
        if not data:
            data, spot_price, data_source = generate_synthetic_data(symbol)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    df = pd.DataFrame(data)
    atm_iv = df[(df['strike'] >= spot_price * 0.99) & (df['strike'] <= spot_price * 1.01)]['iv'].mean() * 100
    
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Spot Price", f"${spot_price:.2f}")
    with col3:
        st.metric("ATM IV", f"{atm_iv:.1f}%")
    with col4:
        status = "🟢 Live" if data_source == "live" else "🟡 Demo"
        st.metric("Data Source", status)
    
    st.markdown("---")
    
    # Main 3D Surface
    st.plotly_chart(
        create_3d_surface(data, spot_price, symbol, data_source),
        use_container_width=True
    )
    
    # Two-column layout for skew and term structure
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.plotly_chart(create_skew_chart(data, spot_price, symbol), use_container_width=True)
    
    with col_right:
        st.plotly_chart(create_term_structure(data, spot_price), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Data points: {len(data)} | 
            Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
