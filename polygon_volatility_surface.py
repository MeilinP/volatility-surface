"""
Live Volatility Surface - Polygon.io Version
=============================================

A real-time implied volatility surface visualization using Polygon.io API.
Based on the Quant Guild IBKR implementation, adapted for Polygon.io data.

Author: Meilin Pan
Original Concept: Roman Paolucci (Quant Guild)

Requirements:
- Polygon.io API key (free tier works but with delays)
- pip install polygon-api-client pandas numpy matplotlib

Usage:
1. Set your POLYGON_API_KEY environment variable or pass it directly
2. Run: python polygon_volatility_surface.py
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

# Polygon.io REST client
from polygon import RESTClient

# Use dark background style for professional look
plt.style.use('dark_background')


class PolygonVolatilitySurface:
    """
    Fetches options data from Polygon.io and builds a live volatility surface.
    
    Polygon.io provides options snapshots including Greeks and IV through their
    Options Snapshot API endpoint.
    """
    
    def __init__(self, api_key: str, symbol: str = 'SPY'):
        """
        Initialize the Polygon volatility surface fetcher.
        
        Parameters:
        -----------
        api_key : str - Your Polygon.io API key
        symbol : str - Underlying ticker symbol (default: SPY)
        """
        self.client = RESTClient(api_key)
        self.symbol = symbol.upper()
        
        # Data storage
        self.spot_price: float = 0.0
        self.iv_data: List[Dict] = []
        self.last_update: Optional[datetime] = None
        
        # Configuration
        self.num_expirations = 6  # Number of expiration dates to fetch
        self.moneyness_range = (0.95, 1.05)  # Strike range as % of spot
        
    def get_spot_price(self) -> float:
        """Fetch the current spot price for the underlying."""
        try:
            # Get previous day's close (most reliable)
            aggs = list(self.client.get_aggs(
                ticker=self.symbol,
                multiplier=1,
                timespan="day",
                from_=datetime.now() - timedelta(days=5),
                to=datetime.now(),
                limit=1,
                sort="desc"
            ))
            
            if aggs:
                self.spot_price = aggs[0].close
                print(f"Spot price for {self.symbol}: ${self.spot_price:.2f}")
                return self.spot_price
            else:
                raise ValueError("No price data returned")
                
        except Exception as e:
            print(f"Error fetching spot price: {e}")
            # Fallback to a reasonable default for SPY
            if self.symbol == 'SPY':
                self.spot_price = 585.0
            return self.spot_price
    
    def get_options_contracts(self) -> List[Dict]:
        """
        Fetch available options contracts from Polygon.
        
        Returns a list of option contract tickers.
        """
        try:
            # Get options contracts for the symbol
            contracts = []
            
            # Calculate target expirations (next N Fridays/monthly expirations)
            today = datetime.now()
            target_exps = []
            
            # Get expirations for the next few months
            for i in range(self.num_expirations + 2):
                # Check weekly and monthly expirations
                check_date = today + timedelta(days=i * 7)
                target_exps.append(check_date.strftime('%Y-%m-%d'))
            
            # Fetch options chain using Polygon's options contracts endpoint
            response = self.client.list_options_contracts(
                underlying_ticker=self.symbol,
                expiration_date_gte=today.strftime('%Y-%m-%d'),
                expiration_date_lte=(today + timedelta(days=180)).strftime('%Y-%m-%d'),
                limit=1000
            )
            
            for contract in response:
                contracts.append({
                    'ticker': contract.ticker,
                    'strike': contract.strike_price,
                    'expiration': contract.expiration_date,
                    'type': contract.contract_type
                })
            
            return contracts
            
        except Exception as e:
            print(f"Error fetching options contracts: {e}")
            return []
    
    def get_options_snapshot(self) -> pd.DataFrame:
        """
        Fetch options snapshot data including IV from Polygon.
        
        The snapshot endpoint provides greeks and IV for all options.
        """
        try:
            # Fetch options chain snapshot
            snapshot = self.client.get_snapshot_option(
                underlying_asset=self.symbol,
                option_contract="all"
            )
            
            data = []
            for option in snapshot:
                # Extract relevant fields
                if hasattr(option, 'implied_volatility') and option.implied_volatility:
                    # Parse the option ticker to get strike and expiration
                    # Polygon option tickers: O:SPY250221C00585000
                    ticker = option.details.ticker if hasattr(option, 'details') else ''
                    
                    data.append({
                        'ticker': ticker,
                        'strike': option.details.strike_price if hasattr(option.details, 'strike_price') else 0,
                        'expiration': option.details.expiration_date if hasattr(option.details, 'expiration_date') else '',
                        'type': option.details.contract_type if hasattr(option.details, 'contract_type') else '',
                        'iv': option.implied_volatility,
                        'delta': option.greeks.delta if hasattr(option, 'greeks') and option.greeks else None,
                        'gamma': option.greeks.gamma if hasattr(option, 'greeks') and option.greeks else None,
                        'theta': option.greeks.theta if hasattr(option, 'greeks') and option.greeks else None,
                        'vega': option.greeks.vega if hasattr(option, 'greeks') and option.greeks else None,
                        'last_price': option.last_quote.midpoint if hasattr(option, 'last_quote') and option.last_quote else None
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error fetching options snapshot: {e}")
            return pd.DataFrame()
    
    def fetch_iv_data_rest(self) -> List[Dict]:
        """
        Fetch IV data using Polygon REST API (alternative method).
        
        This fetches individual option quotes and calculates basic stats.
        """
        if self.spot_price == 0:
            self.get_spot_price()
        
        iv_data = []
        
        try:
            # Get options contracts
            contracts = self.get_options_contracts()
            
            if not contracts:
                print("No contracts found, using snapshot method...")
                return self.fetch_iv_from_snapshot()
            
            # Filter by moneyness
            min_strike = self.spot_price * self.moneyness_range[0]
            max_strike = self.spot_price * self.moneyness_range[1]
            
            filtered = [c for c in contracts 
                       if min_strike <= c['strike'] <= max_strike]
            
            # Group by expiration and get unique expirations
            expirations = sorted(list(set(c['expiration'] for c in filtered)))[:self.num_expirations]
            
            # Fetch IV for each contract (limited to avoid rate limits)
            for contract in filtered[:100]:  # Limit to 100 contracts
                if contract['expiration'] not in expirations:
                    continue
                    
                try:
                    # Get the last quote for this option
                    ticker = contract['ticker']
                    quotes = self.client.get_last_quote(ticker)
                    
                    if quotes:
                        # Note: Polygon provides IV in snapshots, not individual quotes
                        # For individual contracts, we'd need to calculate IV ourselves
                        iv_data.append({
                            'expiration': contract['expiration'],
                            'strike': contract['strike'],
                            'type': contract['type'],
                            'ticker': ticker
                        })
                        
                except Exception as e:
                    continue
                    
                time.sleep(0.1)  # Rate limiting
            
            return iv_data
            
        except Exception as e:
            print(f"Error in REST fetch: {e}")
            return []
    
    def fetch_iv_from_snapshot(self) -> List[Dict]:
        """
        Primary method: Fetch IV data from Polygon's options snapshot.
        """
        if self.spot_price == 0:
            self.get_spot_price()
        
        print(f"Fetching options snapshot for {self.symbol}...")
        
        try:
            # Try the full chain snapshot
            data = []
            
            # Use the options chain endpoint
            chain = self.client.list_snapshot_options_chain(
                underlying_asset=self.symbol,
            )
            
            min_strike = self.spot_price * self.moneyness_range[0]
            max_strike = self.spot_price * self.moneyness_range[1]
            
            for option in chain:
                try:
                    strike = option.details.strike_price
                    
                    # Filter by moneyness
                    if not (min_strike <= strike <= max_strike):
                        continue
                    
                    # Get IV (Polygon provides this in the snapshot)
                    iv = option.implied_volatility if hasattr(option, 'implied_volatility') else None
                    
                    if iv and iv > 0:
                        data.append({
                            'expiration': option.details.expiration_date,
                            'strike': strike,
                            'iv': iv,
                            'type': option.details.contract_type,
                            'delta': option.greeks.delta if option.greeks else None,
                        })
                        
                except Exception as e:
                    continue
            
            # Filter to get the next N expirations
            if data:
                df = pd.DataFrame(data)
                expirations = sorted(df['expiration'].unique())[:self.num_expirations]
                data = [d for d in data if d['expiration'] in expirations]
            
            self.iv_data = data
            self.last_update = datetime.now()
            print(f"Fetched {len(data)} IV data points")
            
            return data
            
        except Exception as e:
            print(f"Snapshot error: {e}")
            print("Falling back to synthetic data for demonstration...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self) -> List[Dict]:
        """
        Generate realistic synthetic IV data for demonstration.
        
        Uses typical market patterns (smile, skew, term structure).
        """
        if self.spot_price == 0:
            self.spot_price = 585.0 if self.symbol == 'SPY' else 100.0
        
        print(f"Generating synthetic data (Spot: ${self.spot_price:.2f})")
        
        data = []
        base_vol = 0.18
        
        # Generate expirations (next 6 dates)
        today = datetime.now()
        expirations = []
        for i in [7, 14, 30, 45, 60, 90]:
            exp_date = today + timedelta(days=i)
            expirations.append(exp_date.strftime('%Y-%m-%d'))
        
        # Generate strikes
        strikes = np.linspace(
            self.spot_price * self.moneyness_range[0],
            self.spot_price * self.moneyness_range[1],
            15
        )
        
        for exp_date in expirations:
            # Calculate DTE
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_dt - today).days
            T = dte / 365.0
            
            for strike in strikes:
                moneyness = strike / self.spot_price
                log_m = np.log(moneyness)
                
                # Realistic IV model
                skew = -0.12 * log_m  # Negative skew
                smile = 0.06 * log_m ** 2  # Smile curvature
                term = 0.02 * np.sqrt(T)  # Term structure
                noise = np.random.normal(0, 0.003)
                
                iv = base_vol + skew + smile + term + noise
                iv = max(0.05, min(1.0, iv))
                
                # Add small time-varying component for "live" feel
                iv += np.random.normal(0, 0.001)
                
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
        """Refresh the IV data from Polygon or synthetic source."""
        try:
            data = self.fetch_iv_from_snapshot()
            if not data:
                data = self.generate_synthetic_data()
            return data
        except Exception as e:
            print(f"Refresh error: {e}")
            return self.generate_synthetic_data()


class PlotState:
    """Manages the UI state for locking/unlocking updates."""
    
    def __init__(self):
        self.is_locked = False
        self.btn_label = None
    
    def toggle(self, event):
        self.is_locked = not self.is_locked
        if self.btn_label:
            self.btn_label.set_text("UNLOCK UPDATES" if self.is_locked else "LOCK UPDATES")
        plt.draw()


def create_surface_grid(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Convert IV data to a grid suitable for 3D plotting.
    
    Returns:
    --------
    X : Strike prices meshgrid
    Y : Expiration index meshgrid
    Z : IV values grid
    exp_labels : Expiration date labels
    """
    df = pd.DataFrame(data)
    
    # Create pivot table: rows = expirations, columns = strikes
    pivot = df.pivot_table(
        index='expiration', 
        columns='strike', 
        values='iv',
        aggfunc='mean'  # Average if multiple (calls & puts)
    ).sort_index().sort_index(axis=1)
    
    # Interpolate missing values
    pivot = pivot.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
    pivot = pivot.bfill().ffill()
    
    # Create meshgrid
    X, Y_idx = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
    Z = pivot.values
    
    return X, Y_idx, Z, list(pivot.index)


def live_surface_plot(surface: PolygonVolatilitySurface, refresh_interval: float = 2.0):
    """
    Create and maintain a live volatility surface visualization.
    
    Parameters:
    -----------
    surface : PolygonVolatilitySurface - Data source
    refresh_interval : float - Seconds between data refreshes
    """
    # Enable interactive mode
    plt.ion()
    
    # Create figure with dark theme
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f'Live Volatility Surface - {surface.symbol}')
    fig.patch.set_facecolor('#0b0d0f')
    
    # Create subplots: 3D surface (left), 2D skew (right)
    ax_3d = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d')
    ax_skew = plt.subplot2grid((1, 3), (0, 2))
    
    # Initialize UI button
    state = PlotState()
    ax_button = plt.axes([0.42, 0.03, 0.12, 0.04])
    btn = Button(ax_button, 'LOCK UPDATES', color='#1f2329', hovercolor='#2d333b')
    state.btn_label = btn.label
    btn.label.set_color('white')
    btn.label.set_fontsize(9)
    btn.on_clicked(state.toggle)
    
    # Add info text
    ax_info = plt.axes([0.02, 0.02, 0.2, 0.04])
    ax_info.set_facecolor('#0b0d0f')
    ax_info.axis('off')
    info_text = ax_info.text(0, 0.5, '', color='white', fontsize=8, 
                             verticalalignment='center')
    
    print(f"\n{'='*60}")
    print(f"  Live Volatility Surface Started - {surface.symbol}")
    print(f"  Spot: ${surface.spot_price:.2f}")
    print(f"  Press Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    try:
        while True:
            if not state.is_locked:
                # Refresh data
                data = surface.refresh_data()
                
                if len(data) > 10:
                    # Create surface grid
                    X, Y_idx, Z, exp_labels = create_surface_grid(data)
                    
                    # Save current camera angle
                    curr_elev = ax_3d.elev if hasattr(ax_3d, 'elev') else 30
                    curr_azim = ax_3d.azim if hasattr(ax_3d, 'azim') else -60
                    
                    # --- Update 3D Surface ---
                    ax_3d.clear()
                    ax_3d.set_facecolor('#0b0d0f')
                    
                    # Plot surface with magma colormap
                    surf = ax_3d.plot_surface(
                        X, Y_idx, Z * 100,  # Convert to percentage
                        cmap='magma',
                        edgecolor='white',
                        linewidth=0.1,
                        alpha=0.9
                    )
                    
                    # Format axes
                    ax_3d.set_xlabel('Strike Price ($)', color='white', fontsize=9)
                    ax_3d.set_ylabel('Expiration', color='white', fontsize=9)
                    ax_3d.set_zlabel('IV (%)', color='white', fontsize=9)
                    
                    # Set expiration labels
                    ax_3d.set_yticks(np.arange(len(exp_labels)))
                    ax_3d.set_yticklabels([e[5:] for e in exp_labels], fontsize=7)  # MM-DD format
                    
                    # Title with timestamp
                    ax_3d.set_title(
                        f"LIVE IV SURFACE | {surface.symbol} | {time.strftime('%H:%M:%S')}",
                        color='white',
                        fontsize=11,
                        fontweight='bold'
                    )
                    
                    # Restore camera angle
                    ax_3d.view_init(elev=curr_elev, azim=curr_azim)
                    
                    # Style the 3D axes
                    ax_3d.xaxis.pane.fill = False
                    ax_3d.yaxis.pane.fill = False
                    ax_3d.zaxis.pane.fill = False
                    ax_3d.tick_params(colors='white', labelsize=8)
                    
                    # --- Update 2D Skew Chart ---
                    ax_skew.clear()
                    ax_skew.set_facecolor('#161b22')
                    
                    # Get front-month data
                    df = pd.DataFrame(data)
                    front_exp = sorted(df['expiration'].unique())[0]
                    skew_data = df[df['expiration'] == front_exp].sort_values('strike')
                    
                    # Plot the skew
                    ax_skew.plot(
                        skew_data['strike'], 
                        skew_data['iv'] * 100,
                        marker='o',
                        color='#00f2ff',
                        linewidth=2,
                        markersize=6
                    )
                    
                    # Add spot price line
                    ax_skew.axvline(
                        x=surface.spot_price,
                        color='#ff3e3e',
                        linestyle='--',
                        linewidth=2,
                        label=f'Spot: ${surface.spot_price:.2f}'
                    )
                    
                    # Format skew chart
                    ax_skew.set_title(
                        f"FRONT-MONTH SKEW: {front_exp}",
                        color='white',
                        fontsize=10,
                        fontweight='bold'
                    )
                    ax_skew.set_xlabel('Strike Price ($)', color='white', fontsize=9)
                    ax_skew.set_ylabel('Implied Volatility (%)', color='white', fontsize=9)
                    ax_skew.tick_params(colors='white', labelsize=8)
                    ax_skew.legend(loc='upper right', fontsize=8)
                    ax_skew.grid(True, alpha=0.3)
                    
                    # Update info text
                    info_text.set_text(
                        f"Data Points: {len(data)} | "
                        f"Last Update: {time.strftime('%H:%M:%S')}"
                    )
            
            # Pause for refresh
            plt.pause(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Live Volatility Surface - Polygon.io Edition")
    print("=" * 60)
    
    # Get API key from environment or prompt
    api_key = os.environ.get('POLYGON_API_KEY', '')
    
    if not api_key:
        print("\nNo POLYGON_API_KEY found in environment.")
        print("Running in DEMO MODE with synthetic data.\n")
        api_key = "DEMO"  # Will trigger synthetic data fallback
    
    # Configuration
    SYMBOL = 'SPY'
    REFRESH_INTERVAL = 2.0  # Seconds between updates
    
    # Initialize the surface
    surface = PolygonVolatilitySurface(api_key=api_key, symbol=SYMBOL)
    
    # Get initial spot price
    surface.get_spot_price()
    
    # Fetch initial data
    print("\nFetching initial IV data...")
    data = surface.refresh_data()
    print(f"Loaded {len(data)} data points\n")
    
    # Start the live visualization
    live_surface_plot(surface, refresh_interval=REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
