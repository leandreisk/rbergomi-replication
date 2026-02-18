import wrds
import pandas as pd
import numpy as np
from scipy import interpolate

class OptionMetricsFetcher:
    def __init__(self):
        """Initializes the WRDS connection."""
        try:
            self.db = wrds.Connection()
        except Exception as e:
            print(f"Connection error: {e}")
            self.db = None

    def close(self):
        if self.db: self.db.close()

    def get_secid(self, ticker):
        """Retrieves the security ID (SECID) for a ticker."""
        if not self.db: return None
        query = f"SELECT secid FROM optionm.secnmd WHERE ticker = '{ticker}' ORDER BY effect_date DESC LIMIT 1"
        try:
            val = self.db.raw_sql(query)
            return val.iloc[0]['secid'] if not val.empty else None
        except: return None

    def get_stock_prices(self, secid, start, end):
        """Fetches daily closing stock prices."""
        if not self.db: return None
        query = f"SELECT date, close as stock_price FROM optionm.secprd WHERE secid = {secid} AND date BETWEEN '{start}' AND '{end}'"
        df = self.db.raw_sql(query)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def get_zero_rates(self, start, end):
        """Fetches the zero-coupon yield curve."""
        if not self.db: return None
        query = f"SELECT date, days, rate FROM optionm.zerocd WHERE date BETWEEN '{start}' AND '{end}'"
        df = self.db.raw_sql(query)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['rate'] = df['rate'] / 100.0  # Convert percent to decimal
        return df

    def get_raw_options(self, secid, start, end):
        """Fetches raw option data for a specific year."""
        if not self.db: return None
        table = f"optionm.opprcd{pd.to_datetime(start).year}"
        query = f"""
            SELECT date, exdate, cp_flag, strike_price, best_bid, best_offer, 
                   volume, open_interest, impl_volatility, delta, gamma, vega, theta
            FROM {table} WHERE secid = {secid} AND date BETWEEN '{start}' AND '{end}'
        """
        try:
            df = self.db.raw_sql(query)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['exdate'] = pd.to_datetime(df['exdate'])
            return df
        except: return None

    def _interpolate_rates(self, df_opts, df_rates):
        """Interpolates risk-free rates matching option maturity."""
        df_opts['risk_free_rate'] = np.nan
        
        for date in df_opts['date'].unique():
            curve = df_rates[df_rates['date'] == date].sort_values('days')
            if curve.empty: continue

            f = interpolate.interp1d(curve['days'], curve['rate'], kind='linear', fill_value="extrapolate")
            mask = df_opts['date'] == date
            # Clip maturity to avoid negative values during interpolation
            df_opts.loc[mask, 'risk_free_rate'] = f(df_opts.loc[mask, 'days_to_maturity'].clip(lower=1))
            
        return df_opts

    def process_data(self, df_opts, df_stock, df_rates):
        """Merges data and calculates derived metrics."""
        if df_opts is None or df_stock is None: return None

        # Basic calculations
        df_opts['strike'] = df_opts['strike_price'] / 1000.0
        df_opts['mid_price'] = 0.5 * (df_opts['best_bid'] + df_opts['best_offer'])
        df_opts['days_to_maturity'] = (df_opts['exdate'] - df_opts['date']).dt.days
        df_opts['years_to_maturity'] = df_opts['days_to_maturity'] / 365.0

        # Rate interpolation
        if df_rates is not None:
            df_opts = self._interpolate_rates(df_opts, df_rates)
        else:
            df_opts['risk_free_rate'] = np.nan

        # Merge and Moneyness
        df = pd.merge(df_opts, df_stock, on='date', how='inner')
        df['moneyness'] = df['stock_price'] / df['strike']

        target_cols = [
            'date', 'exdate', 'stock_price', 'strike', 'cp_flag', 'moneyness',
            'mid_price', 'best_bid', 'best_offer','risk_free_rate', 'impl_volatility', 'delta', 'gamma', 
            'vega', 'theta', 'days_to_maturity', 'years_to_maturity', 'volume', 'open_interest'
        ]
        return df[[c for c in target_cols if c in df.columns]]

    def fetch_data(self, ticker, start, end):
        """Main execution method."""
        secid = self.get_secid(ticker)
        if not secid: return None

        df_stock = self.get_stock_prices(secid, start, end)
        df_rates = self.get_zero_rates(start, end)
        df_opts = self.get_raw_options(secid, start, end)

        return self.process_data(df_opts, df_stock, df_rates)