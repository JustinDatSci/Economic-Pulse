# 🚀 Enhanced Real-Time Data Loader with Multiple APIs
# Advanced data integration with Alpha Vantage, FRED, Yahoo Finance

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

class RealTimeDataLoader:
    """Enhanced real-time data loader with multiple API sources"""
    
    def __init__(self):
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.fred_key = st.secrets.get("FRED_API_KEY", "demo")
        self.cache_duration = 300  # 5 minutes cache
        self.base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'fred': 'https://api.stlouisfed.org/fred/series/observations'
        }
    
    @st.cache_data(ttl=300)
    def get_economic_indicators(_self):
        """Fetch real economic indicators from FRED API"""
        
        indicators = {
            'UNRATE': {'name': 'US Unemployment Rate', 'category': 'Employment'},
            'FEDFUNDS': {'name': 'Federal Funds Rate', 'category': 'Monetary Policy'},
            'CPIAUCSL': {'name': 'Consumer Price Index', 'category': 'Inflation'},
            'GDP': {'name': 'Gross Domestic Product', 'category': 'Growth'},
            'HOUST': {'name': 'Housing Starts', 'category': 'Housing'},
            'INDPRO': {'name': 'Industrial Production Index', 'category': 'Production'},
            'PAYEMS': {'name': 'Nonfarm Payrolls', 'category': 'Employment'},
            'UMCSENT': {'name': 'Consumer Sentiment', 'category': 'Sentiment'},
            'VIXCLS': {'name': 'VIX Volatility Index', 'category': 'Market Risk'}
        }
        
        all_data = []
        
        for series_id, info in indicators.items():
            try:
                # Try FRED API first
                if _self.fred_key != "demo":
                    url = f"{_self.base_urls['fred']}?series_id={series_id}&api_key={_self.fred_key}&file_type=json&limit=60"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'observations' in data:
                            for obs in data['observations']:
                                if obs['value'] != '.':  # Skip missing values
                                    all_data.append({
                                        'date': pd.to_datetime(obs['date']).date(),
                                        'value': float(obs['value']),
                                        'series_id': series_id,
                                        'series_name': info['name'],
                                        'category': info['category'],
                                        'unit': '%' if 'Rate' in info['name'] or series_id in ['UNRATE', 'FEDFUNDS'] else 'Index',
                                        'asset_type': 'Economic',
                                        'source': 'FRED'
                                    })
                            continue
                
                # Fallback to realistic simulated data
                all_data.extend(_self._generate_economic_fallback(series_id, info))
                
            except Exception as e:
                st.warning(f"⚠️ Could not fetch {series_id}: {str(e)}")
                all_data.extend(_self._generate_economic_fallback(series_id, info))
        
        return all_data
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbols=None):
        """Fetch real-time stock data from Yahoo Finance and Alpha Vantage"""
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'VTI', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV']
        
        stock_data = []
        
        for symbol in symbols:
            try:
                # Try Yahoo Finance first
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                
                if not hist.empty:
                    info = ticker.info
                    stock_name = info.get('longName', f'{symbol} ETF')
                    
                    for date, row in hist.iterrows():
                        stock_data.append({
                            'date': date.date(),
                            'value': row['Close'],
                            'volume': row['Volume'],
                            'high': row['High'],
                            'low': row['Low'],
                            'open': row['Open'],
                            'series_id': symbol,
                            'series_name': stock_name,
                            'category': 'Stock Market',
                            'unit': 'USD',
                            'asset_type': 'Equity',
                            'source': 'Yahoo Finance'
                        })
                
                # Try Alpha Vantage as backup
                elif _self.alpha_vantage_key != "demo":
                    url = f"{_self.base_urls['alpha_vantage']}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={_self.alpha_vantage_key}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'Time Series (Daily)' in data:
                            time_series = data['Time Series (Daily)']
                            
                            for date_str, values in list(time_series.items())[-252:]:  # Last year
                                stock_data.append({
                                    'date': pd.to_datetime(date_str).date(),
                                    'value': float(values['4. close']),
                                    'volume': int(values['5. volume']),
                                    'high': float(values['2. high']),
                                    'low': float(values['3. low']),
                                    'open': float(values['1. open']),
                                    'series_id': symbol,
                                    'series_name': f'{symbol} Stock',
                                    'category': 'Stock Market',
                                    'unit': 'USD',
                                    'asset_type': 'Equity',
                                    'source': 'Alpha Vantage'
                                })
                else:
                    # Fallback to simulated data
                    stock_data.extend(_self._generate_stock_fallback(symbol))
                    
            except Exception as e:
                st.warning(f"⚠️ Could not fetch {symbol}: {str(e)}")
                stock_data.extend(_self._generate_stock_fallback(symbol))
        
        return stock_data
    
    @st.cache_data(ttl=300)
    def get_crypto_data(_self, symbols=None):
        """Fetch real-time cryptocurrency data"""
        
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
        
        crypto_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                
                if not hist.empty:
                    crypto_name = symbol.replace('-USD', '').upper()
                    
                    for date, row in hist.iterrows():
                        crypto_data.append({
                            'date': date.date(),
                            'value': row['Close'],
                            'volume': row['Volume'],
                            'high': row['High'],
                            'low': row['Low'],
                            'open': row['Open'],
                            'series_id': symbol,
                            'series_name': f'{crypto_name} Price',
                            'category': 'Cryptocurrency',
                            'unit': 'USD',
                            'asset_type': 'Crypto',
                            'source': 'Yahoo Finance'
                        })
                else:
                    crypto_data.extend(_self._generate_crypto_fallback(symbol))
                    
            except Exception as e:
                st.warning(f"⚠️ Could not fetch {symbol}: {str(e)}")
                crypto_data.extend(_self._generate_crypto_fallback(symbol))
        
        return crypto_data
    
    @st.cache_data(ttl=300)
    def get_forex_data(_self, pairs=None):
        """Fetch real-time forex data"""
        
        if pairs is None:
            pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
        
        forex_data = []
        
        for pair in pairs:
            try:
                ticker = yf.Ticker(pair)
                hist = ticker.history(period='1y')
                
                if not hist.empty:
                    pair_name = pair.replace('=X', '').upper()
                    
                    for date, row in hist.iterrows():
                        forex_data.append({
                            'date': date.date(),
                            'value': row['Close'],
                            'volume': row.get('Volume', 0),
                            'high': row['High'],
                            'low': row['Low'],
                            'open': row['Open'],
                            'series_id': pair,
                            'series_name': f'{pair_name} Exchange Rate',
                            'category': 'Forex',
                            'unit': 'Exchange Rate',
                            'asset_type': 'Currency',
                            'source': 'Yahoo Finance'
                        })
                else:
                    forex_data.extend(_self._generate_forex_fallback(pair))
                    
            except Exception as e:
                st.warning(f"⚠️ Could not fetch {pair}: {str(e)}")
                forex_data.extend(_self._generate_forex_fallback(pair))
        
        return forex_data
    
    def _generate_economic_fallback(self, series_id, info):
        """Generate realistic fallback data for economic indicators"""
        
        dates = pd.date_range('2022-01-01', datetime.now(), freq='M')
        data = []
        
        # Economic-specific patterns
        if series_id == 'UNRATE':
            base_values = np.full(len(dates), 3.7)
            base_values += np.random.normal(0, 0.2, len(dates))
        elif series_id == 'FEDFUNDS':
            base_values = np.linspace(0.25, 5.25, len(dates))
            base_values += np.random.normal(0, 0.1, len(dates))
        elif series_id == 'VIXCLS':
            base_values = 20 + np.random.normal(0, 8, len(dates))
            base_values = np.clip(base_values, 10, 80)
        else:
            trend = 1.02 ** (np.arange(len(dates)) / 12)
            base_values = 100 * trend + np.random.normal(0, 2, len(dates))
        
        for i, date in enumerate(dates):
            data.append({
                'date': date.date(),
                'value': max(0.1, base_values[i]),
                'series_id': series_id,
                'series_name': info['name'],
                'category': info['category'],
                'unit': '%' if 'Rate' in info['name'] or series_id in ['UNRATE', 'FEDFUNDS'] else 'Index',
                'asset_type': 'Economic',
                'source': 'Simulated'
            })
        
        return data
    
    def _generate_stock_fallback(self, symbol):
        """Generate realistic fallback stock data"""
        
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_prices = {'SPY': 450, 'QQQ': 380, 'IWM': 200, 'GLD': 180, 'VTI': 240, 'TLT': 90}
        base_price = base_prices.get(symbol, 100)
        
        prices = [base_price]
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.0008, 0.015)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(1.0, new_price))
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date.date(),
                'value': prices[i],
                'volume': np.random.randint(1000000, 10000000),
                'high': prices[i] * 1.02,
                'low': prices[i] * 0.98,
                'open': prices[i] * 0.999,
                'series_id': symbol,
                'series_name': f'{symbol} ETF',
                'category': 'Stock Market',
                'unit': 'USD',
                'asset_type': 'Equity',
                'source': 'Simulated'
            })
        
        return data
    
    def _generate_crypto_fallback(self, symbol):
        """Generate realistic fallback crypto data"""
        
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_prices = {'BTC-USD': 45000, 'ETH-USD': 3000, 'BNB-USD': 300, 'ADA-USD': 0.5, 'SOL-USD': 100}
        base_price = base_prices.get(symbol, 1000)
        
        prices = [base_price]
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.001, 0.04)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(0.01, new_price))
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date.date(),
                'value': prices[i],
                'volume': np.random.randint(100000, 1000000),
                'high': prices[i] * 1.05,
                'low': prices[i] * 0.95,
                'open': prices[i] * 0.998,
                'series_id': symbol,
                'series_name': f'{symbol.replace("-USD", "")} Price',
                'category': 'Cryptocurrency',
                'unit': 'USD',
                'asset_type': 'Crypto',
                'source': 'Simulated'
            })
        
        return data
    
    def _generate_forex_fallback(self, pair):
        """Generate realistic fallback forex data"""
        
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_rates = {'EURUSD=X': 1.08, 'GBPUSD=X': 1.25, 'USDJPY=X': 150.0, 'AUDUSD=X': 0.65, 'USDCAD=X': 1.35}
        base_rate = base_rates.get(pair, 1.0)
        
        rates = [base_rate]
        for i in range(1, len(dates)):
            daily_change = np.random.normal(0, 0.005)
            new_rate = rates[-1] * (1 + daily_change)
            rates.append(max(0.01, new_rate))
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date.date(),
                'value': rates[i],
                'volume': np.random.randint(1000000, 10000000),
                'high': rates[i] * 1.01,
                'low': rates[i] * 0.99,
                'open': rates[i] * 0.999,
                'series_id': pair,
                'series_name': f'{pair.replace("=X", "")} Exchange Rate',
                'category': 'Forex',
                'unit': 'Exchange Rate',
                'asset_type': 'Currency',
                'source': 'Simulated'
            })
        
        return data
    
    def load_all_real_time_data(self):
        """Load comprehensive real-time data from all sources"""
        
        all_data = []
        
        # Economic indicators
        all_data.extend(self.get_economic_indicators())
        
        # Stock market data
        all_data.extend(self.get_stock_data())
        
        # Cryptocurrency data
        all_data.extend(self.get_crypto_data())
        
        # Forex data
        all_data.extend(self.get_forex_data())
        
        return pd.DataFrame(all_data)

# Real-time data refresh utilities
class DataRefreshManager:
    """Manage real-time data refresh and caching"""
    
    def __init__(self):
        self.last_refresh = {}
        self.refresh_intervals = {
            'economic': 3600,  # 1 hour
            'stock': 300,      # 5 minutes
            'crypto': 60,      # 1 minute
            'forex': 300       # 5 minutes
        }
    
    def needs_refresh(self, data_type):
        """Check if data needs refresh"""
        
        if data_type not in self.last_refresh:
            return True
        
        time_since_refresh = time.time() - self.last_refresh[data_type]
        return time_since_refresh > self.refresh_intervals.get(data_type, 300)
    
    def mark_refreshed(self, data_type):
        """Mark data as refreshed"""
        self.last_refresh[data_type] = time.time()
    
    def get_refresh_status(self):
        """Get refresh status for all data types"""
        
        status = {}
        for data_type in self.refresh_intervals:
            if data_type in self.last_refresh:
                time_since = time.time() - self.last_refresh[data_type]
                status[data_type] = {
                    'last_refresh': datetime.fromtimestamp(self.last_refresh[data_type]),
                    'needs_refresh': self.needs_refresh(data_type),
                    'time_until_refresh': max(0, self.refresh_intervals[data_type] - time_since)
                }
            else:
                status[data_type] = {
                    'last_refresh': None,
                    'needs_refresh': True,
                    'time_until_refresh': 0
                }
        
        return status

# API status checker
def check_api_status():
    """Check status of all external APIs"""
    
    status = {
        'yahoo_finance': False,
        'alpha_vantage': False,
        'fred': False
    }
    
    try:
        # Test Yahoo Finance
        test_ticker = yf.Ticker('SPY')
        test_data = test_ticker.history(period='1d')
        status['yahoo_finance'] = not test_data.empty
    except:
        pass
    
    try:
        # Test Alpha Vantage
        alpha_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        if alpha_key != "demo":
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey={alpha_key}"
            response = requests.get(url, timeout=5)
            status['alpha_vantage'] = response.status_code == 200
    except:
        pass
    
    try:
        # Test FRED
        fred_key = st.secrets.get("FRED_API_KEY", "demo")
        if fred_key != "demo":
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={fred_key}&file_type=json&limit=1"
            response = requests.get(url, timeout=5)
            status['fred'] = response.status_code == 200
    except:
        pass
    
    return status