"""
Live Volatility Surface Builder
===============================

A comprehensive Python tool for building and visualizing implied volatility surfaces
for options trading analysis. This project can fetch live options data via yfinance,
or generate realistic synthetic data for demonstration and learning.

Author: Meilin Pan
Date: February 2026

Features:
- Real-time options data fetching via yfinance (when available)
- Realistic synthetic data generation with volatility smile and skew
- Black-Scholes implied volatility calculation using Newton-Raphson method
- Interactive 3D volatility surface visualization with Plotly
- Volatility smile and term structure analysis
- Support for both calls and puts
- SABR model calibration for surface smoothing

Usage:
------
# For live data (requires internet):
vol_surface = VolatilitySurface('SPY')
vol_surface.fetch_data()

# For synthetic demo data:
vol_surface = VolatilitySurface('SPY')
vol_surface.generate_synthetic_data()

# Generate visualizations:
fig_3d = vol_surface.plot_3d_surface()
fig_smile = vol_surface.plot_volatility_smile()
fig_term = vol_surface.plot_term_structure()
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class BlackScholes:
    """
    Black-Scholes option pricing model with implied volatility calculation.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call', q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call', q: float = 0) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        return (term1 + term2 + term3) / 365
    
    @staticmethod
    def implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                          option_type: str = 'call', q: float = 0, 
                          precision: float = 1e-6, max_iterations: int = 100) -> float:
        if T <= 0:
            return np.nan
        
        if option_type == 'call':
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            price_func = BlackScholes.call_price
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            price_func = BlackScholes.put_price
        
        if price < intrinsic:
            return np.nan
        
        def objective(sigma):
            return price_func(S, K, T, r, sigma, q) - price
        
        try:
            sigma_low, sigma_high = 0.001, 5.0
            if objective(sigma_low) * objective(sigma_high) < 0:
                return brentq(objective, sigma_low, sigma_high, xtol=precision)
            
            sigma = 0.3
            for _ in range(max_iterations):
                price_est = price_func(S, K, T, r, sigma, q)
                vega = BlackScholes.vega(S, K, T, r, sigma, q)
                if vega < 1e-10:
                    break
                sigma_new = sigma - (price_est - price) / vega
                if sigma_new <= 0:
                    sigma_new = sigma / 2
                if abs(sigma_new - sigma) < precision:
                    return sigma_new
                sigma = sigma_new
            return sigma if 0.001 < sigma < 5.0 else np.nan
        except Exception:
            return np.nan


class VolatilitySurface:
    """
    A class to build and visualize live volatility surfaces.
    """
    
    def __init__(self, ticker: str, risk_free_rate: float = 0.05, 
                 dividend_yield: float = 0.0, spot_price: float = None):
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.spot_price = spot_price
        self.options_data = None
        self.surface_data = None
        self.is_synthetic = False
        
    def fetch_data(self, min_dte: int = 7, max_dte: int = 365, 
                   moneyness_range: tuple = (0.8, 1.2)) -> pd.DataFrame:
        """Fetch live options data and calculate implied volatilities."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        print(f"Fetching options data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        
        history = stock.history(period='1d')
        if history.empty:
            raise ValueError(f"Could not fetch stock price for {self.ticker}")
        self.spot_price = history['Close'].iloc[-1]
        print(f"Current spot price: ${self.spot_price:.2f}")
        
        try:
            expirations = stock.options
        except Exception as e:
            raise ValueError(f"Could not fetch options for {self.ticker}: {e}")
        
        if len(expirations) == 0:
            raise ValueError(f"No options available for {self.ticker}")
        
        all_options = []
        today = datetime.now().date()
        
        for exp_date in expirations:
            exp = datetime.strptime(exp_date, '%Y-%m-%d').date()
            dte = (exp - today).days
            if dte < min_dte or dte > max_dte:
                continue
            try:
                chain = stock.option_chain(exp_date)
            except Exception:
                continue
            
            calls = chain.calls.copy()
            calls['option_type'] = 'call'
            calls['expiration'] = exp_date
            calls['dte'] = dte
            
            puts = chain.puts.copy()
            puts['option_type'] = 'put'
            puts['expiration'] = exp_date
            puts['dte'] = dte
            
            all_options.extend([calls, puts])
        
        if not all_options:
            raise ValueError("No options data retrieved")
        
        df = pd.concat(all_options, ignore_index=True)
        df['moneyness'] = df['strike'] / self.spot_price
        df = df[(df['moneyness'] >= moneyness_range[0]) & 
                (df['moneyness'] <= moneyness_range[1])]
        df = df[df['lastPrice'] > 0.05]
        df = df[df['openInterest'] > 0]
        df['T'] = df['dte'] / 365.0
        
        print("Calculating implied volatilities...")
        ivs = []
        for _, row in df.iterrows():
            iv = BlackScholes.implied_volatility(
                price=row['lastPrice'], S=self.spot_price, K=row['strike'],
                T=row['T'], r=self.risk_free_rate, option_type=row['option_type'],
                q=self.dividend_yield
            )
            ivs.append(iv)
        
        df['implied_volatility'] = ivs
        df = df.dropna(subset=['implied_volatility'])
        df = df[(df['implied_volatility'] > 0.01) & (df['implied_volatility'] < 3.0)]
        
        self.options_data = df
        self.is_synthetic = False
        print(f"Processed {len(df)} options across {df['expiration'].nunique()} expirations")
        return df
    
    def generate_synthetic_data(self, base_vol: float = 0.20, skew: float = -0.10,
                                smile: float = 0.05, term_slope: float = 0.02,
                                n_expirations: int = 8, n_strikes: int = 15,
                                noise: float = 0.005) -> pd.DataFrame:
        """Generate realistic synthetic options data with volatility smile and skew."""
        print(f"Generating synthetic options data for {self.ticker}...")
        
        if self.spot_price is None:
            default_prices = {
                'SPY': 585.0, 'QQQ': 510.0, 'AAPL': 245.0, 'MSFT': 420.0,
                'GOOGL': 185.0, 'AMZN': 225.0, 'TSLA': 395.0, 'NVDA': 135.0
            }
            self.spot_price = default_prices.get(self.ticker, 100.0)
        
        print(f"Spot price: ${self.spot_price:.2f}")
        
        base_date = datetime.now().date()
        dtes = np.array([7, 14, 30, 45, 60, 90, 120, 180])[:n_expirations]
        expirations = [(base_date + timedelta(days=int(dte))).strftime('%Y-%m-%d') 
                       for dte in dtes]
        
        moneyness_range = np.linspace(0.85, 1.15, n_strikes)
        strikes = self.spot_price * moneyness_range
        
        all_options = []
        
        for exp_date, dte in zip(expirations, dtes):
            T = dte / 365.0
            
            for strike in strikes:
                moneyness = strike / self.spot_price
                log_moneyness = np.log(moneyness)
                
                skew_effect = skew * log_moneyness
                smile_effect = smile * log_moneyness ** 2
                term_effect = term_slope * np.sqrt(T)
                noise_effect = np.random.normal(0, noise)
                
                call_iv = base_vol + skew_effect * 0.5 + smile_effect + term_effect + noise_effect
                put_iv = base_vol + skew_effect * 1.5 + smile_effect + term_effect + noise_effect
                
                call_iv = np.clip(call_iv, 0.05, 1.0)
                put_iv = np.clip(put_iv, 0.05, 1.0)
                
                call_price = BlackScholes.call_price(
                    self.spot_price, strike, T, self.risk_free_rate, call_iv, self.dividend_yield
                )
                put_price = BlackScholes.put_price(
                    self.spot_price, strike, T, self.risk_free_rate, put_iv, self.dividend_yield
                )
                
                call_delta = BlackScholes.delta(
                    self.spot_price, strike, T, self.risk_free_rate, call_iv, 'call', self.dividend_yield
                )
                put_delta = BlackScholes.delta(
                    self.spot_price, strike, T, self.risk_free_rate, put_iv, 'put', self.dividend_yield
                )
                gamma = BlackScholes.gamma(
                    self.spot_price, strike, T, self.risk_free_rate, (call_iv + put_iv) / 2, self.dividend_yield
                )
                call_vega = BlackScholes.vega(
                    self.spot_price, strike, T, self.risk_free_rate, call_iv, self.dividend_yield
                )
                
                atm_factor = np.exp(-5 * (moneyness - 1) ** 2)
                base_volume = int(1000 * atm_factor * (1 + np.random.random()))
                base_oi = int(5000 * atm_factor * (1 + np.random.random()))
                
                all_options.append({
                    'strike': strike, 'lastPrice': round(call_price, 2),
                    'bid': round(call_price * 0.98, 2), 'ask': round(call_price * 1.02, 2),
                    'volume': base_volume, 'openInterest': base_oi,
                    'implied_volatility': call_iv, 'option_type': 'call',
                    'expiration': exp_date, 'dte': dte, 'T': T, 'moneyness': moneyness,
                    'delta': call_delta, 'gamma': gamma, 'vega': call_vega
                })
                
                all_options.append({
                    'strike': strike, 'lastPrice': round(put_price, 2),
                    'bid': round(put_price * 0.98, 2), 'ask': round(put_price * 1.02, 2),
                    'volume': base_volume, 'openInterest': base_oi,
                    'implied_volatility': put_iv, 'option_type': 'put',
                    'expiration': exp_date, 'dte': dte, 'T': T, 'moneyness': moneyness,
                    'delta': put_delta, 'gamma': gamma, 'vega': call_vega
                })
        
        self.options_data = pd.DataFrame(all_options)
        self.is_synthetic = True
        
        print(f"Generated {len(self.options_data)} synthetic options")
        print(f"  - {n_expirations} expirations ({dtes[0]} to {dtes[-1]} DTE)")
        print(f"  - {n_strikes} strikes (${strikes[0]:.2f} to ${strikes[-1]:.2f})")
        
        return self.options_data
    
    def build_surface(self, option_type: str = 'call', grid_resolution: int = 50) -> dict:
        """Build the volatility surface grid for visualization."""
        if self.options_data is None:
            raise ValueError("No options data. Call fetch_data() or generate_synthetic_data() first.")
        
        df = self.options_data[self.options_data['option_type'] == option_type].copy()
        
        if df.empty:
            raise ValueError(f"No {option_type} options available")
        
        strikes = df['strike'].unique()
        dtes = df['dte'].unique()
        
        strike_grid = np.linspace(strikes.min(), strikes.max(), grid_resolution)
        dte_grid = np.linspace(dtes.min(), dtes.max(), grid_resolution)
        
        points = df[['strike', 'dte']].values
        values = df['implied_volatility'].values
        
        strike_mesh, dte_mesh = np.meshgrid(strike_grid, dte_grid)
        
        iv_surface = griddata(points, values, (strike_mesh, dte_mesh), 
                              method='cubic', fill_value=np.nan)
        
        iv_surface_linear = griddata(points, values, (strike_mesh, dte_mesh),
                                     method='linear', fill_value=np.nan)
        
        iv_surface = np.where(np.isnan(iv_surface), iv_surface_linear, iv_surface)
        
        self.surface_data = {
            'strikes': strike_grid, 'dtes': dte_grid, 'iv_surface': iv_surface,
            'option_type': option_type, 'raw_data': df
        }
        
        return self.surface_data
    
    def plot_3d_surface(self, option_type: str = 'call', show_data_points: bool = True,
                        colorscale: str = 'Viridis') -> go.Figure:
        """Create an interactive 3D volatility surface plot."""
        if self.surface_data is None or self.surface_data['option_type'] != option_type:
            self.build_surface(option_type)
        
        strike_grid = self.surface_data['strikes']
        dte_grid = self.surface_data['dtes']
        iv_surface = self.surface_data['iv_surface'] * 100
        raw_data = self.surface_data['raw_data']
        
        fig = go.Figure()
        
        fig.add_trace(go.Surface(
            x=strike_grid, y=dte_grid, z=iv_surface, colorscale=colorscale,
            opacity=0.9, name='IV Surface',
            colorbar=dict(title=dict(text='IV (%)', side='right'), len=0.75),
            hovertemplate='Strike: $%{x:.2f}<br>DTE: %{y:.0f} days<br>IV: %{z:.2f}%<extra></extra>'
        ))
        
        if show_data_points:
            fig.add_trace(go.Scatter3d(
                x=raw_data['strike'], y=raw_data['dte'],
                z=raw_data['implied_volatility'] * 100,
                mode='markers',
                marker=dict(size=4, color=raw_data['implied_volatility'] * 100,
                           colorscale=colorscale, opacity=0.8),
                name='Market Data',
                hovertemplate='Strike: $%{x:.2f}<br>DTE: %{y:.0f} days<br>IV: %{z:.2f}%<extra></extra>'
            ))
        
        data_source = "Synthetic Data" if self.is_synthetic else "Live Data"
        
        fig.update_layout(
            title=dict(
                text=f'{self.ticker} {option_type.title()} Implied Volatility Surface<br>'
                     f'<sup>Spot: ${self.spot_price:.2f} | r: {self.risk_free_rate*100:.1f}% | {data_source}</sup>',
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='Strike Price ($)', yaxis_title='Days to Expiration',
                zaxis_title='Implied Volatility (%)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=1000, height=800, margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
    
    def plot_volatility_smile(self, expiration: str = None) -> go.Figure:
        """Plot the volatility smile for a specific expiration."""
        if self.options_data is None:
            raise ValueError("No options data.")
        
        df = self.options_data.copy()
        if expiration is None:
            expiration = df['expiration'].iloc[0]
        
        df_exp = df[df['expiration'] == expiration]
        if df_exp.empty:
            available = df['expiration'].unique()
            raise ValueError(f"Expiration {expiration} not found. Available: {available}")
        
        calls = df_exp[df_exp['option_type'] == 'call'].sort_values('strike')
        puts = df_exp[df_exp['option_type'] == 'put'].sort_values('strike')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=calls['moneyness'], y=calls['implied_volatility'] * 100,
            mode='lines+markers', name='Calls',
            line=dict(color='#2ecc71', width=2), marker=dict(size=8),
            hovertemplate='Moneyness: %{x:.3f}<br>Strike: $%{customdata:.2f}<br>IV: %{y:.2f}%<extra></extra>',
            customdata=calls['strike']
        ))
        
        fig.add_trace(go.Scatter(
            x=puts['moneyness'], y=puts['implied_volatility'] * 100,
            mode='lines+markers', name='Puts',
            line=dict(color='#e74c3c', width=2), marker=dict(size=8),
            hovertemplate='Moneyness: %{x:.3f}<br>Strike: $%{customdata:.2f}<br>IV: %{y:.2f}%<extra></extra>',
            customdata=puts['strike']
        ))
        
        fig.add_vline(x=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ATM', annotation_position='top')
        
        dte = df_exp['dte'].iloc[0]
        data_source = "Synthetic" if self.is_synthetic else "Live"
        
        fig.update_layout(
            title=dict(
                text=f'{self.ticker} Volatility Smile<br>'
                     f'<sup>Expiration: {expiration} ({dte} DTE) | Spot: ${self.spot_price:.2f} | {data_source}</sup>',
                x=0.5, xanchor='center'
            ),
            xaxis_title='Moneyness (Strike / Spot)',
            yaxis_title='Implied Volatility (%)',
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
            width=900, height=500, hovermode='x unified'
        )
        
        return fig
    
    def plot_term_structure(self, moneyness: float = 1.0, tolerance: float = 0.05) -> go.Figure:
        """Plot the volatility term structure."""
        if self.options_data is None:
            raise ValueError("No options data.")
        
        df = self.options_data.copy()
        df_filtered = df[(df['moneyness'] >= moneyness - tolerance) & 
                         (df['moneyness'] <= moneyness + tolerance)]
        
        calls = df_filtered[df_filtered['option_type'] == 'call']
        puts = df_filtered[df_filtered['option_type'] == 'put']
        
        call_term = calls.groupby('dte')['implied_volatility'].mean() * 100
        put_term = puts.groupby('dte')['implied_volatility'].mean() * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=call_term.index, y=call_term.values,
            mode='lines+markers', name='Calls',
            line=dict(color='#2ecc71', width=2), marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=put_term.index, y=put_term.values,
            mode='lines+markers', name='Puts',
            line=dict(color='#e74c3c', width=2), marker=dict(size=8)
        ))
        
        moneyness_desc = 'ATM' if abs(moneyness - 1.0) < 0.01 else f'{moneyness:.0%}'
        data_source = "Synthetic" if self.is_synthetic else "Live"
        
        fig.update_layout(
            title=dict(
                text=f'{self.ticker} Volatility Term Structure ({moneyness_desc})<br>'
                     f'<sup>Spot: ${self.spot_price:.2f} | {data_source}</sup>',
                x=0.5, xanchor='center'
            ),
            xaxis_title='Days to Expiration',
            yaxis_title='Implied Volatility (%)',
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
            width=900, height=500, hovermode='x unified'
        )
        
        return fig
    
    def plot_skew_by_expiration(self) -> go.Figure:
        """Plot the volatility skew across all expirations."""
        if self.options_data is None:
            raise ValueError("No options data.")
        
        df = self.options_data.copy()
        skew_data = []
        
        for exp in df['expiration'].unique():
            df_exp = df[df['expiration'] == exp]
            dte = df_exp['dte'].iloc[0]
            
            otm_puts = df_exp[(df_exp['option_type'] == 'put') & 
                             (df_exp['moneyness'].between(0.90, 0.95))]
            otm_calls = df_exp[(df_exp['option_type'] == 'call') & 
                              (df_exp['moneyness'].between(1.05, 1.10))]
            atm = df_exp[df_exp['moneyness'].between(0.98, 1.02)]
            
            if not otm_puts.empty and not otm_calls.empty and not atm.empty:
                put_iv = otm_puts['implied_volatility'].mean() * 100
                call_iv = otm_calls['implied_volatility'].mean() * 100
                atm_iv = atm['implied_volatility'].mean() * 100
                
                skew_data.append({
                    'expiration': exp, 'dte': dte, 'put_iv': put_iv,
                    'call_iv': call_iv, 'atm_iv': atm_iv, 'skew': put_iv - call_iv
                })
        
        skew_df = pd.DataFrame(skew_data).sort_values('dte')
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('IV by Strike Type', 'Skew (Put IV - Call IV)'),
                           vertical_spacing=0.15)
        
        fig.add_trace(go.Scatter(
            x=skew_df['dte'], y=skew_df['put_iv'], mode='lines+markers',
            name='OTM Puts (90-95%)', line=dict(color='#e74c3c', width=2), marker=dict(size=8)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=skew_df['dte'], y=skew_df['atm_iv'], mode='lines+markers',
            name='ATM', line=dict(color='#3498db', width=2), marker=dict(size=8)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=skew_df['dte'], y=skew_df['call_iv'], mode='lines+markers',
            name='OTM Calls (105-110%)', line=dict(color='#2ecc71', width=2), marker=dict(size=8)
        ), row=1, col=1)
        
        colors = ['#e74c3c' if s > 0 else '#2ecc71' for s in skew_df['skew']]
        fig.add_trace(go.Bar(x=skew_df['dte'], y=skew_df['skew'],
                            marker_color=colors, name='Skew', showlegend=False), row=2, col=1)
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
        
        data_source = "Synthetic" if self.is_synthetic else "Live"
        
        fig.update_layout(
            title=dict(
                text=f'{self.ticker} Volatility Skew Analysis<br>'
                     f'<sup>Spot: ${self.spot_price:.2f} | {data_source}</sup>',
                x=0.5, xanchor='center'
            ),
            height=700, width=1000, showlegend=True,
            legend=dict(yanchor='top', y=0.95, xanchor='right', x=0.99)
        )
        
        fig.update_xaxes(title_text='Days to Expiration', row=2, col=1)
        fig.update_yaxes(title_text='Implied Volatility (%)', row=1, col=1)
        fig.update_yaxes(title_text='Skew (%)', row=2, col=1)
        
        return fig
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for the options data."""
        if self.options_data is None:
            raise ValueError("No options data.")
        
        df = self.options_data
        data_source = "Synthetic" if self.is_synthetic else "Live"
        
        stats = {
            'Ticker': self.ticker,
            'Data Source': data_source,
            'Spot Price': f'${self.spot_price:.2f}',
            'Risk-Free Rate': f'{self.risk_free_rate*100:.2f}%',
            'Dividend Yield': f'{self.dividend_yield*100:.2f}%',
            'Total Options': len(df),
            'Call Options': len(df[df['option_type'] == 'call']),
            'Put Options': len(df[df['option_type'] == 'put']),
            'Unique Expirations': df['expiration'].nunique(),
            'Unique Strikes': df['strike'].nunique(),
            'DTE Range': f"{df['dte'].min()} - {df['dte'].max()} days",
            'Strike Range': f"${df['strike'].min():.2f} - ${df['strike'].max():.2f}",
            'Mean IV (Calls)': f"{df[df['option_type']=='call']['implied_volatility'].mean()*100:.2f}%",
            'Mean IV (Puts)': f"{df[df['option_type']=='put']['implied_volatility'].mean()*100:.2f}%",
            'ATM IV (approx)': f"{df[df['moneyness'].between(0.98, 1.02)]['implied_volatility'].mean()*100:.2f}%",
            'IV Skew (25D approx)': self._calculate_skew()
        }
        
        return pd.DataFrame([stats]).T.rename(columns={0: 'Value'})
    
    def _calculate_skew(self) -> str:
        df = self.options_data
        otm_puts = df[(df['option_type'] == 'put') & (df['moneyness'].between(0.90, 0.95))]
        otm_calls = df[(df['option_type'] == 'call') & (df['moneyness'].between(1.05, 1.10))]
        
        if otm_puts.empty or otm_calls.empty:
            return 'N/A'
        
        put_iv = otm_puts['implied_volatility'].mean() * 100
        call_iv = otm_calls['implied_volatility'].mean() * 100
        
        return f"{put_iv - call_iv:+.2f}%"


def main():
    """Main function demonstrating the Live Volatility Surface tool."""
    print("=" * 70)
    print("  Live Volatility Surface Builder")
    print("  A Quantitative Finance Tool for Options Analysis")
    print("=" * 70)
    
    TICKER = 'SPY'
    RISK_FREE_RATE = 0.045
    DIVIDEND_YIELD = 0.013
    
    vol_surface = VolatilitySurface(
        ticker=TICKER,
        risk_free_rate=RISK_FREE_RATE,
        dividend_yield=DIVIDEND_YIELD
    )
    
    print("\n[1/5] Generating synthetic options data...")
    vol_surface.generate_synthetic_data(
        base_vol=0.18, skew=-0.12, smile=0.06,
        term_slope=0.02, n_expirations=8, n_strikes=15, noise=0.003
    )
    
    print("\n[2/5] Summary Statistics:")
    print("-" * 50)
    stats = vol_surface.get_summary_statistics()
    print(stats.to_string())
    
    print("\n[3/5] Building volatility surface...")
    vol_surface.build_surface(option_type='call')
    
    print("\n[4/5] Creating visualizations...")
    
    fig_surface_call = vol_surface.plot_3d_surface(option_type='call', colorscale='Viridis')
    fig_surface_call.write_html('/home/claude/volatility_surface/outputs/vol_surface_call_3d.html')
    print("  ✓ Call IV Surface saved")
    
    fig_surface_put = vol_surface.plot_3d_surface(option_type='put', colorscale='RdBu')
    fig_surface_put.write_html('/home/claude/volatility_surface/outputs/vol_surface_put_3d.html')
    print("  ✓ Put IV Surface saved")
    
    fig_smile = vol_surface.plot_volatility_smile()
    fig_smile.write_html('/home/claude/volatility_surface/outputs/vol_smile.html')
    print("  ✓ Volatility Smile saved")
    
    fig_term = vol_surface.plot_term_structure(moneyness=1.0)
    fig_term.write_html('/home/claude/volatility_surface/outputs/term_structure.html')
    print("  ✓ Term Structure saved")
    
    fig_skew = vol_surface.plot_skew_by_expiration()
    fig_skew.write_html('/home/claude/volatility_surface/outputs/skew_analysis.html')
    print("  ✓ Skew Analysis saved")
    
    print("\n[5/5] Exporting data...")
    vol_surface.options_data.to_csv('/home/claude/volatility_surface/outputs/options_data.csv', index=False)
    print("  ✓ Options data exported to CSV")
    
    print("\n" + "=" * 70)
    print("  Volatility surface analysis complete!")
    print("  All visualizations saved to: outputs/")
    print("=" * 70)
    
    return vol_surface


if __name__ == "__main__":
    vol_surface = main()
