"""
Live Volatility Surface - yfinance
Real-time implied volatility surface visualization.
"""

import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import yfinance as yf

plt.style.use('dark_background')


class VolatilitySurface:
    def __init__(self, symbol: str = 'SPY'):
        self.symbol = symbol.upper()
        self.spot_price = 0.0
        self.iv_data = []
        self.num_expirations = 6
        self.moneyness_range = (0.95, 1.05)

    def get_spot_price(self) -> float:
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                self.spot_price = hist['Close'].iloc[-1]
        except Exception as e:
            print(f"Price error: {e}")
            self.spot_price = 600.0

        print(f"Spot: ${self.spot_price:.2f}")
        return self.spot_price

    def fetch_iv_data(self) -> List[Dict]:
        if not self.spot_price:
            self.get_spot_price()

        min_strike = self.spot_price * self.moneyness_range[0]
        max_strike = self.spot_price * self.moneyness_range[1]
        data = []

        try:
            ticker = yf.Ticker(self.symbol)
            expirations = ticker.options[:self.num_expirations]

            for exp in expirations:
                chain = ticker.option_chain(exp)
                for _, row in chain.calls.iterrows():
                    strike = row.get('strike')
                    iv = row.get('impliedVolatility')
                    if iv and strike and 0.05 < iv < 0.80 and min_strike <= strike <= max_strike:
                        data.append({
                            'expiration': exp,
                            'strike': strike,
                            'iv': iv,
                            'type': 'call'
                        })

            self.iv_data = data
            print(f"Fetched {len(data)} contracts")
            return data

        except Exception as e:
            print(f"Error: {e}")
            return []


def build_surface(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index='expiration', columns='strike', values='iv', aggfunc='mean'
    ).sort_index().sort_index(axis=1)

    pivot = pivot.interpolate(axis=0).interpolate(axis=1).bfill().ffill()
    X, Y = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))

    return X, Y, pivot.values, list(pivot.index)


def run(symbol: str = 'SPY', interval: float = 2.0):
    surface = VolatilitySurface(symbol)
    surface.get_spot_price()

    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f'IV Surface - {symbol}')
    fig.patch.set_facecolor('#0b0d0f')

    ax_3d = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d')
    ax_skew = plt.subplot2grid((1, 3), (0, 2))

    locked = [False]
    ax_btn = plt.axes([0.42, 0.03, 0.12, 0.04])
    btn = Button(ax_btn, 'LOCK', color='#1f2329', hovercolor='#2d333b')
    btn.label.set_color('white')
    btn.on_clicked(lambda _: locked.__setitem__(0, not locked[0]) or
                   btn.label.set_text('UNLOCK' if locked[0] else 'LOCK'))

    print(f"\n{'='*50}\n  Live IV Surface - {symbol}\n  Ctrl+C to exit\n{'='*50}\n")

    try:
        while True:
            if not locked[0]:
                surface.get_spot_price()
                data = surface.fetch_iv_data()

                if len(data) > 10:
                    X, Y, Z, exps = build_surface(data)
                    elev, azim = ax_3d.elev or 30, ax_3d.azim or -60

                    ax_3d.clear()
                    ax_3d.set_facecolor('#0b0d0f')
                    ax_3d.plot_surface(X, Y, Z * 100, cmap='magma', edgecolor='white', lw=0.1, alpha=0.9)
                    ax_3d.set_xlabel('Strike ($)', color='white')
                    ax_3d.set_ylabel('Expiration', color='white')
                    ax_3d.set_zlabel('IV (%)', color='white')
                    ax_3d.set_yticks(range(len(exps)))
                    ax_3d.set_yticklabels([e[5:] for e in exps], fontsize=7)
                    ax_3d.set_title(
                        f"IV SURFACE | {symbol} @ ${surface.spot_price:.2f} | {time.strftime('%H:%M:%S')}",
                        color='white'
                    )
                    ax_3d.view_init(elev, azim)
                    ax_3d.tick_params(colors='white', labelsize=8)

                    ax_skew.clear()
                    ax_skew.set_facecolor('#161b22')
                    df = pd.DataFrame(data)
                    front = df[df['expiration'] == sorted(df['expiration'].unique())[0]].sort_values('strike')
                    ax_skew.plot(front['strike'], front['iv'] * 100, 'o-', color='#00f2ff', lw=2, ms=4)
                    ax_skew.axvline(surface.spot_price, color='#ff3e3e', ls='--', lw=2)
                    ax_skew.set_title(f"SKEW: {sorted(df['expiration'].unique())[0]}", color='white')
                    ax_skew.set_xlabel('Strike ($)', color='white')
                    ax_skew.set_ylabel('IV (%)', color='white')
                    ax_skew.tick_params(colors='white')
                    ax_skew.grid(True, alpha=0.3)

            plt.pause(interval)

    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    run()
