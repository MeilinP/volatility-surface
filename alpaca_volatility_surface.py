"""
Live Volatility Surface - Alpaca Markets Version
=================================================

A real-time implied volatility surface visualization using Alpaca Markets API.
Alpaca now offers options trading and market data through their API.

Author: Meilin Pan
Original Concept: Roman Paolucci (Quant Guild)

Requirements:
- Alpaca account with options enabled
- pip install alpaca-py pandas numpy matplotlib scipy

Usage:
1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
2. Run: python alpaca_volatility_surface.py

Note: Alpaca's options API requires a brokerage account with options enabled.
For paper trading, use the paper trading endpoint.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from scipy.stats import norm
from scipy.optimize import brentq

# Use dark background style
plt.style.use('dark_background')


# =============================================================================
# Black-Scholes for IV Calculation (when IV not provided by API)
# =============================================================================

class BlackScholesIV:
    """Calculate implied volatility from option prices."""
    
    @staticmethod
    def bs_price(S, K, T, r, sigma, option_type='call'):
        """Black-Scholes option price."""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type='call'):
        """Calculate IV using Brent's method."""
        if T <= 0 or price <= 0:
            return np.nan
        
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        if price < intrinsic:
            return np.nan
        
        def objective(sigma):
            return BlackScholesIV.bs_price(S, K, T, r, sigma, option_type) - price
        
        try:
            return brentq(objective, 0.001, 5.0, xtol=1e-6)
        except:
            return np.nan


# =============================================================================
# Alpaca Volatility Surface
# =============================================================================

class AlpacaVolatilitySurface:
    """
    Fetches options data from Alpaca and builds a live volatility surface.
    
    Alpaca provides options market data through their trading API.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True, symbol: str = 'SPY'):
        """
        Initialize the Alpaca volatility surface fetcher.
        
        Parameters:
        -----------
        api_key : str - Your Alpaca API key
        secret_key : str - Your Alpaca secret key
        paper : bool - Use paper trading endpoint (default: True)
        symbol : str - Underlying ticker symbol (default: SPY)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.symbol = symbol.upper()
        
        # API endpoints
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
        # Data storage
        self.spot_price: float = 0.0
        self.iv_data: List[Dict] = []
        self.last_update: Optional[datetime] = None
        self.risk_free_rate = 0.045  # Current approximate risk-free rate
        
        # Configuration
        self.num_expirations = 6
        self.moneyness_range = (0.95, 1.05)
        
        # Initialize Alpaca client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Alpaca API client."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            
            # Store imports for later use
            self.StockLatestQuoteRequest = StockLatestQuoteRequest
            self.StockBarsRequest = StockBarsRequest
            self.TimeFrame = TimeFrame
            
            self.client_initialized = True
            print(f"Alpaca client initialized ({'Paper' if self.paper else 'Live'} trading)")
            
        except ImportError:
            print("Warning: alpaca-py not installed. Using demo mode.")
            self.client_initialized = False
        except Exception as e:
            print(f"Warning: Could not initialize Alpaca client: {e}")
            self.client_initialized = False
    
    def get_spot_price(self) -> float:
        """Fetch the current spot price for the underlying."""
        if not self.client_initialized:
            return self._get_fallback_spot()
        
        try:
            # Get latest quote
            request = self.StockLatestQuoteRequest(symbol_or_symbols=self.symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if self.symbol in quotes:
                quote = quotes[self.symbol]
                # Use midpoint of bid/ask
                self.spot_price = (quote.bid_price + quote.ask_price) / 2
                print(f"Spot price for {self.symbol}: ${self.spot_price:.2f}")
                return self.spot_price
            
        except Exception as e:
            print(f"Error fetching spot price: {e}")
        
        return self._get_fallback_spot()
    
    def _get_fallback_spot(self) -> float:
        """Return fallback spot prices for common tickers."""
        defaults = {
            'SPY': 585.0, 'QQQ': 510.0, 'AAPL': 245.0,
            'MSFT': 420.0, 'NVDA': 135.0, 'TSLA': 395.0
        }
        self.spot_price = defaults.get(self.symbol, 100.0)
        print(f"Using fallback spot for {self.symbol}: ${self.spot_price:.2f}")
        return self.spot_price
    
    def fetch_options_chain(self) -> List[Dict]:
        """
        Fetch options chain from Alpaca.
        
        Note: Alpaca's options API is relatively new. This attempts to use it
        but falls back to synthetic data if not available.
        """
        if not self.client_initialized:
            return []
        
        try:
            # Try to use Alpaca options API
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import AssetStatus
            
            request = GetOptionContractsRequest(
                underlying_symbols=[self.symbol],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=datetime.now().strftime('%Y-%m-%d'),
                expiration_date_lte=(datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            
            options_data = []
            for contract in contracts:
                options_data.append({
                    'symbol': contract.symbol,
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date.strftime('%Y-%m-%d'),
                    'type': contract.type.value,
                    'underlying': contract.underlying_symbol
                })
            
            return options_data
            
        except Exception as e:
            print(f"Options chain fetch error: {e}")
            return []
    
    def fetch_options_quotes(self, contracts: List[Dict]) -> List[Dict]:
        """
        Fetch quotes for options contracts and calculate IV.
        """
        if self.spot_price == 0:
            self.get_spot_price()
        
        iv_data = []
        
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionLatestQuoteRequest
            
            option_client = OptionHistoricalDataClient(self.api_key, self.secret_key)
            
            # Filter contracts by moneyness
            min_strike = self.spot_price * self.moneyness_range[0]
            max_strike = self.spot_price * self.moneyness_range[1]
            
            filtered = [c for c in contracts if min_strike <= c['strike'] <= max_strike]
            
            # Get unique expirations
            expirations = sorted(list(set(c['expiration'] for c in filtered)))[:self.num_expirations]
            filtered = [c for c in filtered if c['expiration'] in expirations]
            
            # Fetch quotes in batches
            symbols = [c['symbol'] for c in filtered[:100]]  # Limit to avoid rate limits
            
            if symbols:
                request = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes = option_client.get_option_latest_quote(request)
                
                for contract in filtered:
                    if contract['symbol'] in quotes:
                        quote = quotes[contract['symbol']]
                        mid_price = (quote.bid_price + quote.ask_price) / 2
                        
                        if mid_price > 0:
                            # Calculate days to expiration
                            exp_date = datetime.strptime(contract['expiration'], '%Y-%m-%d')
                            T = (exp_date - datetime.now()).days / 365.0
                            
                            # Calculate IV
                            iv = BlackScholesIV.implied_volatility(
                                price=mid_price,
                                S=self.spot_price,
                                K=contract['strike'],
                                T=T,
                                r=self.risk_free_rate,
                                option_type=contract['type']
                            )
                            
                            if not np.isnan(iv) and 0.01 < iv < 2.0:
                                iv_data.append({
                                    'expiration': contract['expiration'],
                                    'strike': contract['strike'],
                                    'iv': iv,
                                    'type': contract['type'],
                                    'price': mid_price
                                })
            
            return iv_data
            
        except Exception as e:
            print(f"Options quotes error: {e}")
            return []
    
    def generate_synthetic_data(self) -> List[Dict]:
        """
        Generate realistic synthetic IV data for demonstration.
        """
        if self.spot_price == 0:
            self.get_spot_price()
        
        print(f"Generating synthetic IV data (Spot: ${self.spot_price:.2f})")
        
        data = []
        base_vol = 0.18
        
        # Generate expirations
        today = datetime.now()
        expirations = []
        for days in [7, 14, 30, 45, 60, 90]:
            exp_date = today + timedelta(days=days)
            expirations.append(exp_date.strftime('%Y-%m-%d'))
        
        # Generate strikes
        strikes = np.linspace(
            self.spot_price * self.moneyness_range[0],
            self.spot_price * self.moneyness_range[1],
            15
        )
        
        for exp_date in expirations:
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_dt - today).days
            T = dte / 365.0
            
            for strike in strikes:
                moneyness = strike / self.spot_price
                log_m = np.log(moneyness)
                
                # Realistic IV surface model
                skew = -0.12 * log_m  # Negative skew
                smile = 0.06 * log_m ** 2  # Smile
                term = 0.02 * np.sqrt(T)  # Term structure
                noise = np.random.normal(0, 0.003)
                
                iv = base_vol + skew + smile + term + noise
                iv = max(0.05, min(1.0, iv))
                
                # Add time-varying noise for "live" feel
                iv += np.random.normal(0, 0.002)
                
                data.append({
                    'expiration': exp_date,
                    'strike': round(strike, 2),
                    'iv': iv,
                    'type': 'call' if strike >= self.spot_price else 'put'
                })
        
        self.iv_data = data
        self.last_update = datetime.now()
        
        return data
    
    def refresh_data(self) -> List[Dict]:
        """Refresh IV data from Alpaca or synthetic source."""
        try:
            # Try to fetch real data
            contracts = self.fetch_options_chain()
            
            if contracts:
                iv_data = self.fetch_options_quotes(contracts)
                if iv_data:
                    self.iv_data = iv_data
                    self.last_update = datetime.now()
                    return iv_data
            
            # Fallback to synthetic
            return self.generate_synthetic_data()
            
        except Exception as e:
            print(f"Refresh error: {e}")
            return self.generate_synthetic_data()


# =============================================================================
# Visualization
# =============================================================================

class PlotState:
    """Manages UI state for locking/unlocking updates."""
    
    def __init__(self):
        self.is_locked = False
        self.btn_label = None
    
    def toggle(self, event):
        self.is_locked = not self.is_locked
        if self.btn_label:
            self.btn_label.set_text("UNLOCK UPDATES" if self.is_locked else "LOCK UPDATES")
        plt.draw()


def create_surface_grid(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Convert IV data to a grid for 3D plotting."""
    df = pd.DataFrame(data)
    
    pivot = df.pivot_table(
        index='expiration',
        columns='strike',
        values='iv',
        aggfunc='mean'
    ).sort_index().sort_index(axis=1)
    
    pivot = pivot.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
    pivot = pivot.bfill().ffill()
    
    X, Y_idx = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
    Z = pivot.values
    
    return X, Y_idx, Z, list(pivot.index)


def live_surface_plot(surface: AlpacaVolatilitySurface, refresh_interval: float = 2.0):
    """Create and maintain a live volatility surface visualization."""
    plt.ion()
    
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f'Live Volatility Surface - {surface.symbol} (Alpaca)')
    fig.patch.set_facecolor('#0b0d0f')
    
    ax_3d = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d')
    ax_skew = plt.subplot2grid((1, 3), (0, 2))
    
    state = PlotState()
    ax_button = plt.axes([0.42, 0.03, 0.12, 0.04])
    btn = Button(ax_button, 'LOCK UPDATES', color='#1f2329', hovercolor='#2d333b')
    state.btn_label = btn.label
    btn.label.set_color('white')
    btn.label.set_fontsize(9)
    btn.on_clicked(state.toggle)
    
    ax_info = plt.axes([0.02, 0.02, 0.25, 0.04])
    ax_info.set_facecolor('#0b0d0f')
    ax_info.axis('off')
    info_text = ax_info.text(0, 0.5, '', color='white', fontsize=8, verticalalignment='center')
    
    print(f"\n{'='*60}")
    print(f"  Live Volatility Surface - Alpaca Edition")
    print(f"  Symbol: {surface.symbol} | Spot: ${surface.spot_price:.2f}")
    print(f"  Press Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    try:
        while True:
            if not state.is_locked:
                data = surface.refresh_data()
                
                if len(data) > 10:
                    X, Y_idx, Z, exp_labels = create_surface_grid(data)
                    
                    curr_elev = ax_3d.elev if hasattr(ax_3d, 'elev') else 30
                    curr_azim = ax_3d.azim if hasattr(ax_3d, 'azim') else -60
                    
                    # Update 3D surface
                    ax_3d.clear()
                    ax_3d.set_facecolor('#0b0d0f')
                    
                    ax_3d.plot_surface(
                        X, Y_idx, Z * 100,
                        cmap='magma',
                        edgecolor='white',
                        linewidth=0.1,
                        alpha=0.9
                    )
                    
                    ax_3d.set_xlabel('Strike ($)', color='white', fontsize=9)
                    ax_3d.set_ylabel('Expiration', color='white', fontsize=9)
                    ax_3d.set_zlabel('IV (%)', color='white', fontsize=9)
                    
                    ax_3d.set_yticks(np.arange(len(exp_labels)))
                    ax_3d.set_yticklabels([e[5:] for e in exp_labels], fontsize=7)
                    
                    ax_3d.set_title(
                        f"LIVE IV SURFACE | {surface.symbol} | {time.strftime('%H:%M:%S')}",
                        color='white', fontsize=11, fontweight='bold'
                    )
                    
                    ax_3d.view_init(elev=curr_elev, azim=curr_azim)
                    ax_3d.tick_params(colors='white', labelsize=8)
                    
                    # Update 2D skew
                    ax_skew.clear()
                    ax_skew.set_facecolor('#161b22')
                    
                    df = pd.DataFrame(data)
                    front_exp = sorted(df['expiration'].unique())[0]
                    skew_data = df[df['expiration'] == front_exp].sort_values('strike')
                    
                    ax_skew.plot(
                        skew_data['strike'],
                        skew_data['iv'] * 100,
                        marker='o', color='#00f2ff', linewidth=2, markersize=6
                    )
                    
                    ax_skew.axvline(
                        x=surface.spot_price, color='#ff3e3e',
                        linestyle='--', linewidth=2,
                        label=f'Spot: ${surface.spot_price:.2f}'
                    )
                    
                    ax_skew.set_title(f"FRONT-MONTH SKEW: {front_exp}", color='white', fontsize=10, fontweight='bold')
                    ax_skew.set_xlabel('Strike ($)', color='white', fontsize=9)
                    ax_skew.set_ylabel('IV (%)', color='white', fontsize=9)
                    ax_skew.tick_params(colors='white', labelsize=8)
                    ax_skew.legend(loc='upper right', fontsize=8)
                    ax_skew.grid(True, alpha=0.3)
                    
                    info_text.set_text(
                        f"Data Points: {len(data)} | Spot: ${surface.spot_price:.2f} | "
                        f"Update: {time.strftime('%H:%M:%S')}"
                    )
            
            plt.pause(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Live Volatility Surface - Alpaca Edition")
    print("=" * 60)
    
    # Get API credentials
    api_key = os.environ.get('ALPACA_API_KEY', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
    
    if not api_key or not secret_key:
        print("\nNo Alpaca credentials found in environment.")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        print("Running in DEMO MODE with synthetic data.\n")
        api_key = "DEMO"
        secret_key = "DEMO"
    
    SYMBOL = 'SPY'
    REFRESH_INTERVAL = 2.0
    
    surface = AlpacaVolatilitySurface(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
        symbol=SYMBOL
    )
    
    surface.get_spot_price()
    
    print("\nFetching initial IV data...")
    data = surface.refresh_data()
    print(f"Loaded {len(data)} data points\n")
    
    live_surface_plot(surface, refresh_interval=REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
