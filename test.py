"""
Polygon API Diagnostic - Run this to see actual response structure
"""
import os
from polygon import RESTClient

api_key = "Mx5p8lbXd6gK55pGHvYzBSIHWv0SGMBW"
if not api_key:
    raise ValueError("Set POLYGON_API_KEY")

client = RESTClient(api_key)
chain = list(client.list_snapshot_options_chain("SPY"))

print(f"Total contracts: {len(chain)}")
print(f"\n{'='*60}")
print("FIRST CONTRACT STRUCTURE:")
print(f"{'='*60}")

opt = chain[0]
print(f"\nType: {type(opt)}")
print(f"\nAll attributes: {dir(opt)}")

print(f"\n--- opt.details ---")
print(f"strike_price: {opt.details.strike_price}")
print(f"expiration_date: {opt.details.expiration_date}")
print(f"contract_type: {opt.details.contract_type}")

print(f"\n--- IV locations ---")
print(f"opt.implied_volatility: {getattr(opt, 'implied_volatility', 'NOT FOUND')}")

if hasattr(opt, 'greeks') and opt.greeks:
    print(f"opt.greeks: {opt.greeks}")
    print(f"opt.greeks.iv: {getattr(opt.greeks, 'iv', 'NOT FOUND')}")
else:
    print("opt.greeks: NOT FOUND or None")

print(f"\n--- Underlying asset ---")
if hasattr(opt, 'underlying_asset') and opt.underlying_asset:
    print(f"opt.underlying_asset: {opt.underlying_asset}")
    print(f"opt.underlying_asset.price: {getattr(opt.underlying_asset, 'price', 'NOT FOUND')}")
else:
    print("opt.underlying_asset: NOT FOUND or None")

print(f"\n{'='*60}")
print("SAMPLE OF 10 CONTRACTS WITH IV:")
print(f"{'='*60}")

count = 0
for opt in chain:
    iv = getattr(opt, 'implied_volatility', None)
    greeks_iv = None
    if hasattr(opt, 'greeks') and opt.greeks:
        greeks_iv = getattr(opt.greeks, 'iv', None)
    
    if iv or greeks_iv:
        print(f"Strike: {opt.details.strike_price:>7.2f} | implied_volatility: {str(iv):>10} | greeks.iv: {str(greeks_iv):>10}")
        count += 1
        if count >= 10:
            break

print(f"\n{'='*60}")
print("IV VALUE STATISTICS:")
print(f"{'='*60}")

ivs = []
for opt in chain:
    iv = getattr(opt, 'implied_volatility', None)
    if iv and iv > 0:
        ivs.append(iv)

if ivs:
    print(f"Count with implied_volatility: {len(ivs)}")
    print(f"Min: {min(ivs)}")
    print(f"Max: {max(ivs)}")
    print(f"Mean: {sum(ivs)/len(ivs):.4f}")
    print(f"\nSample values: {ivs[:20]}")
else:
    print("No implied_volatility values found")

greeks_ivs = []
for opt in chain:
    if hasattr(opt, 'greeks') and opt.greeks:
        iv = getattr(opt.greeks, 'iv', None)
        if iv and iv > 0:
            greeks_ivs.append(iv)

if greeks_ivs:
    print(f"\nCount with greeks.iv: {len(greeks_ivs)}")
    print(f"Min: {min(greeks_ivs)}")
    print(f"Max: {max(greeks_ivs)}")
    print(f"Mean: {sum(greeks_ivs)/len(greeks_ivs):.4f}")
    print(f"\nSample values: {greeks_ivs[:20]}")
else:
    print("\nNo greeks.iv values found")