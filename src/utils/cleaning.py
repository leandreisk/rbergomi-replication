import pandas as pd
import numpy as np

def clean_and_filter_options(df):
    """Cleans data and filters for liquid, OTM options with >7 days maturity."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df['cp_flag'] = df['cp_flag'].str.upper().str.strip()

    # Filter 1: Liquidity and Validity
    mask_liq = (
        (df['best_bid'] > 0) & 
        (df['impl_volatility'].between(0.01, 2.0)) & 
        (df['moneyness'].between(0.70, 1.3)) &
        (df['open_interest'] > 10) &
        (df['volume'] > 5) &
        (df['stock_price'] > 0) &
        (df['strike'] > 0)
    )
    df = df[mask_liq].copy()

    # Filter 2: OTM Logic (Calls: Strike >= Spot, Puts: Strike <= Spot)
    mask_call = (df['cp_flag'] == 'C') & (df['strike'] >= df['stock_price'])
    mask_put  = (df['cp_flag'] == 'P') & (df['strike'] <= df['stock_price'])
    df = df[mask_call | mask_put]

    # Filter 3: Minimum Maturity (7 days)
    return df[df['years_to_maturity'] > (7.0 / 365.0)].copy()

def get_calibration_snapshot(df, target_date, target_maturity, tolerance=0.05):
    """Extracts the option chain closest to a specific maturity for a given date."""
    snapshot = df[df['date'] == pd.to_datetime(target_date)].copy()
    
    if snapshot.empty:
        raise ValueError(f"No data found for date {target_date}")

    # Find closest maturity
    maturities = snapshot['years_to_maturity'].unique()
    closest = maturities[np.abs(maturities - target_maturity).argmin()]

    if abs(closest - target_maturity) > tolerance:
        print(f"Warning: Closest maturity ({closest:.4f}) deviates from target ({target_maturity})")

    data = snapshot[snapshot['years_to_maturity'] == closest].sort_values('strike')
    
    return data, closest, snapshot['stock_price'].iloc[0], snapshot["risk_free_rate"].iloc[0]
