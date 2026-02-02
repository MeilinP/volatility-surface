import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from polygon import RESTClient

def fetch_starter_plan_data(api_key, symbol):
    client = RESTClient(api_key)
    try:
        # 1. Fetch the Options Chain (Included in your plan)
        # Note: This will be 15-minute delayed data
        chain = list(client.list_snapshot_options_chain(symbol))
        
        if not chain:
            return None, None, "No data returned"

        # 2. Extract the UNDERLYING price from the options packet
        # This is the only way to get the SPY price without a Stocks plan
        try:
            spot = float(chain[0].underlying_asset.value)
        except (AttributeError, IndexError):
            # Fallback if the field is missing
            st.error("Could not find underlying price in options data.")
            return None, None, "Price Missing"

        data = []
        for opt in chain:
            strike = float(opt.details.strike_price)
            iv = getattr(opt, 'implied_volatility', None)
            is_call = opt.details.contract_type == 'call'
            
            # ONLY use Out-of-the-Money (OTM) to fix the 'fake/jagged' look
            is_otm = (is_call and strike >= spot) or (not is_call and strike < spot)
            
            # Only keep liquid-looking data
            if is_otm and iv and 0.05 < iv < 1.0:
                data.append({
                    'expiration': opt.details.expiration_date,
                    'strike': strike,
                    'iv': iv
                })

        return data, spot, "15-Min Delayed"
    except Exception as e:
        return None, None, str(e)

def main():
    st.title("📊 SPY Volatility Surface (Options Starter)")
    st.info("Plan: Options Starter ($29/mo) | Data: 15-Minute Delayed")
    
    api_key = st.sidebar.text_input("Polygon API Key", type="password")
    
    if api_key:
        # SPY is the only symbol usually available on some starter plans
        data, spot, source = fetch_starter_plan_data(api_key, "SPY")
        
        if data:
            df = pd.DataFrame(data)
            
            # PIVOT & CLEAN
            pivot = df.pivot_table(index='expiration', columns='strike', values='iv', aggfunc='median')
            
            # Remove spikes using a median filter across strikes
            pivot = pivot.T.rolling(window=3, center=True, min_periods=1).median().T
            
            # Fill gaps so the surface is solid
            pivot = pivot.interpolate(method='linear', axis=1).ffill().bfill()

            # PLOT
            strikes, exps = pivot.columns.values, pivot.index.tolist()
            X, Y = np.meshgrid(strikes, np.arange(len(exps)))
            Z = pivot.values * 100

            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Plasma')])
            
            # Add the Red ATM Line
            fig.add_trace(go.Scatter3d(
                x=[spot]*len(exps), y=list(range(len(exps))), 
                z=pivot.values[:, np.abs(strikes - spot).argmin()] * 100,
                mode='lines', line=dict(color='red', width=10), name=f"Spot: {spot}"
            ))

            fig.update_layout(
                title=f"SPY Surface | Underlying: ${spot:.2f} (Delayed)",
                scene=dict(xaxis_title='Strike', yaxis_title='Expiry', zaxis_title='IV %'),
                template="plotly_dark", height=800
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Error: {source}")

if __name__ == "__main__":
    main()
