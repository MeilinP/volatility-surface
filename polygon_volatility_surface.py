import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polygon import RESTClient

class VolatilitySurface:
    def __init__(self, api_key: str, symbol: str = 'SPY'):
        self.client = RESTClient(api_key)
        self.symbol = symbol.upper()
        self.spot_price = 0.0

    def update_spot_price(self) -> float:
        try:
            # Use snapshot for real-time price
            snapshot = self.client.get_snapshot_ticker("stocks", self.symbol)
            if snapshot and hasattr(snapshot, 'last_trade'):
                self.spot_price = snapshot.last_trade.price
                return self.spot_price
        except:
            pass
        return self.spot_price

    def fetch_iv_data(self) -> List[Dict]:
        self.update_spot_price()
        data = []
        try:
            chain = self.client.list_snapshot_options_chain(self.symbol)
            for opt in chain:
                strike = opt.details.strike_price
                iv = getattr(opt, 'implied_volatility', None)
                is_call = opt.details.contract_type == 'call'
                
                # OTM Filtering
                is_otm = (is_call and strike >= self.spot_price) or (not is_call and strike < self.spot_price)
                
                if is_otm and iv and 0.01 < iv < 1.0:
                    data.append({
                        'expiration': opt.details.expiration_date,
                        'strike': strike,
                        'iv': iv
                    })
            
            # Keep first 6 expirations
            exps = sorted(set(d['expiration'] for d in data))[:6]
            return [d for d in data if d['expiration'] in exps]
        except Exception as e:
            print(f"Fetch error: {e}")
            return []

def run_live_view(api_key: str, symbol: str = 'SPY'):
    engine = VolatilitySurface(api_key, symbol)
    plt.ion()
    fig = plt.figure(figsize=(12, 7), facecolor='#0b0d0f')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0b0d0f')

    while True:
        data = engine.fetch_iv_data()
        if data:
            df = pd.DataFrame(data)
            pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='mean')
            
            # Smoothing and Interpolation
            pivot = pivot.rolling(window=3, axis=1, center=True, min_periods=1).mean()
            pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
            
            X, Y = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
            Z = pivot.values * 100
            
            ax.clear()
            ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)
            ax.set_title(f"LIVE {symbol} IV Surface | Spot: ${engine.spot_price:.2f}", color='white')
            ax.set_zlabel("IV %", color='white')
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([e[5:] for e in pivot.index], color='white', fontsize=8)
            ax.tick_params(colors='white')
            
            plt.draw()
            plt.pause(5)
        else:
            print("Waiting for data...")
            time.sleep(5)

if __name__ == "__main__":
    key = os.environ.get('POLYGON_API_KEY')
    if not key: print("Error: Set POLYGON_API_KEY env var")
    else: run_live_view(key)
