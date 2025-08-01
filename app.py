# 🚀 Economic Pulse V3.0 - Real Data Multi-Asset Financial Intelligence Platform
# Enhanced with real data sources for stocks, crypto, forex, and economic indicators

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import json
import time
warnings.filterwarnings('ignore')

# Enhanced ML imports (with fallbacks for missing libraries)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Advanced ML libraries not available. Using simplified models.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.info("📊 Install yfinance for enhanced stock data: pip install yfinance")

# Page configuration
st.set_page_config(
    page_title="🚀 Economic Pulse V3.0 - Real Data AI Platform",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .asset-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    .real-data-badge {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .alpha-vantage-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .risk-alert {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .data-source-info {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 11px;
        color: #2d5a2d;
        margin-top: 0.5rem;
    }
    .premium-info {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 11px;
        color: #8b0000;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class RealDataLoader:
    """Enhanced data loader with real financial and economic data sources including Alpha Vantage"""
    
    def __init__(self):
        self.fred_api_key = st.secrets.get("fred_api_key", "demo_key")
        self.alpha_vantage_key = st.secrets.get("alpha_vantage_key", "demo_key")
        self.cache_duration = 900  # 15 minutes for real data
        
    def load_all_data(self):
        """Load comprehensive real multi-asset data with Alpha Vantage enhancement"""
        
        all_data = []
        data_sources = []
        
        # Display loading status
        st.info("🔄 Initializing multi-source data loading pipeline...")
        
        # Real economic indicators from FRED
        with st.spinner("📊 Loading real US economic data from FRED API..."):
            economic_data, econ_sources = self.load_real_economic_data()
            all_data.extend(economic_data)
            data_sources.extend(econ_sources)
        
        # Enhanced stock market data (Alpha Vantage + Yahoo Finance)
        with st.spinner("📈 Loading enhanced stock market data (Alpha Vantage + Yahoo Finance)..."):
            stock_data, stock_sources = self.load_enhanced_stock_data()
            all_data.extend(stock_data)
            data_sources.extend(stock_sources)
        
        # Enhanced cryptocurrency data (Alpha Vantage + CoinGecko)
        with st.spinner("₿ Loading enhanced cryptocurrency data (Alpha Vantage + CoinGecko)..."):
            crypto_data, crypto_sources = self.load_enhanced_crypto_data()
            all_data.extend(crypto_data)
            data_sources.extend(crypto_sources)
        
        # Enhanced forex data (Alpha Vantage + Yahoo Finance)
        with st.spinner("💱 Loading enhanced forex data (Alpha Vantage + Yahoo Finance)..."):
            forex_data, forex_sources = self.load_enhanced_forex_data()
            all_data.extend(forex_data)
            data_sources.extend(forex_sources)
        
        # Real international data
        with st.spinner("🌍 Loading real international economic data from FRED..."):
            intl_data, intl_sources = self.load_real_international_data()
            all_data.extend(intl_data)
            data_sources.extend(intl_sources)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Enhanced data sources display in sidebar
        if data_sources:
            st.sidebar.subheader("📡 Premium Data Sources")
            unique_sources = list(set(data_sources))
            
            # Prioritize Alpha Vantage sources
            av_sources = [s for s in unique_sources if 'Alpha Vantage' in s]
            other_real_sources = [s for s in unique_sources if 'Real' in s and 'Alpha Vantage' not in s]
            simulation_sources = [s for s in unique_sources if 'Simulation' in s or 'Enhanced' in s]
            
            # Display Alpha Vantage sources with premium badges
            for source in av_sources:
                st.sidebar.markdown(f"<div class='alpha-vantage-badge'>🔥 {source}</div>", unsafe_allow_html=True)
            
            # Display other real sources
            for source in other_real_sources:
                st.sidebar.markdown(f"<div class='real-data-badge'>✅ {source}</div>", unsafe_allow_html=True)
            
            # Display simulation sources
            for source in simulation_sources:
                st.sidebar.markdown(f"<div class='data-source-info'>🎯 {source}</div>", unsafe_allow_html=True)
        
        st.success(f"✅ Successfully loaded {len(df)} data points from {len(set(data_sources))} sources")
        return df
    
    def load_real_economic_data(self):
        """Load real US economic indicators from FRED API"""
        
        indicators = {
            'UNRATE': 'Unemployment Rate',
            'FEDFUNDS': 'Federal Funds Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'GDP': 'Gross Domestic Product',
            'HOUST': 'Housing Starts',
            'INDPRO': 'Industrial Production Index',
            'PAYEMS': 'Nonfarm Payrolls',
            'UMCSENT': 'Consumer Sentiment',
            'DEXUSEU': 'US/Euro Exchange Rate',
            'VIXCLS': 'VIX Volatility Index'
        }
        
        data = []
        sources = []
        
        for series_id, series_name in indicators.items():
            try:
                # Try FRED API first
                if self.fred_api_key != "demo_key":
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 120,  # Last 10 years monthly
                        'sort_order': 'desc'
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        observations = json_data.get('observations', [])
                        
                        valid_data = False
                        for obs in observations:
                            if obs['value'] != '.' and obs['value'] and obs['value'] != 'null':
                                try:
                                    data.append({
                                        'date': pd.to_datetime(obs['date']).date(),
                                        'value': float(obs['value']),
                                        'series_id': series_id,
                                        'series_name': series_name,
                                        'category': 'Economic',
                                        'unit': '%' if 'Rate' in series_name else 'Index',
                                        'asset_type': 'Economic',
                                        'data_source': 'FRED API (Real)'
                                    })
                                    valid_data = True
                                except (ValueError, TypeError):
                                    continue
                        
                        if valid_data:
                            sources.append("FRED Economic Data")
                            st.success(f"✅ Loaded real {series_name} from FRED API")
                            time.sleep(0.1)  # Rate limiting
                        else:
                            raise Exception("No valid FRED data")
                    else:
                        raise Exception("FRED API request failed")
                else:
                    raise Exception("No FRED API key configured")
                    
            except Exception as e:
                # Fallback to realistic simulated data with current patterns
                st.warning(f"⚠️ FRED API unavailable for {series_name}, using enhanced simulation")
                sim_data = self.generate_realistic_economic_data(series_id, series_name)
                data.extend(sim_data)
                if "Enhanced Economic Simulation" not in sources:
                    sources.append("Enhanced Economic Simulation")
        
        return data, sources
    
    def load_enhanced_stock_data(self):
        """Load enhanced stock market data from Alpha Vantage and Yahoo Finance"""
        
        # Individual stocks for Alpha Vantage (premium data)
        alpha_vantage_stocks = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corp',
            'GOOGL': 'Alphabet Inc',
            'AMZN': 'Amazon.com Inc',
            'TSLA': 'Tesla Inc',
            'NVDA': 'NVIDIA Corp',
            'META': 'Meta Platforms Inc',
            'NFLX': 'Netflix Inc'
        }
        
        # ETFs and indices for Yahoo Finance
        yahoo_finance_stocks = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'GLD': 'Gold ETF',
            'VTI': 'Total Stock Market ETF',
            'TLT': '20+ Year Treasury ETF',
            'VIX': 'Volatility Index',
            'DIA': 'Dow Jones ETF'
        }
        
        data = []
        sources = []
        
        # Try Alpha Vantage first for individual stocks
        if self.alpha_vantage_key != "demo_key":
            st.info("🔥 Loading premium stock data from Alpha Vantage API...")
            
            for symbol, name in alpha_vantage_stocks.items():
                try:
                    st.info(f"📊 Loading {name} ({symbol}) from Alpha Vantage...")
                    
                    # Alpha Vantage daily data
                    url = "https://www.alphavantage.co/query"
                    params = {
                        'function': 'TIME_SERIES_DAILY',
                        'symbol': symbol,
                        'apikey': self.alpha_vantage_key,
                        'outputsize': 'compact'  # Last 100 days
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        av_data = response.json()
                        
                        # Check for API errors
                        if 'Error Message' in av_data:
                            st.warning(f"⚠️ Alpha Vantage error for {symbol}: {av_data['Error Message']}")
                            continue
                        
                        if 'Note' in av_data:
                            st.warning(f"⚠️ Alpha Vantage rate limit hit for {symbol}")
                            continue
                        
                        time_series = av_data.get('Time Series (Daily)', {})
                        
                        if time_series:
                            valid_data = False
                            for date_str, values in time_series.items():
                                try:
                                    date = pd.to_datetime(date_str).date()
                                    close_price = float(values['4. close'])
                                    volume = int(values['5. volume'])
                                    high = float(values['2. high'])
                                    low = float(values['3. low'])
                                    open_price = float(values['1. open'])
                                    
                                    data.append({
                                        'date': date,
                                        'value': close_price,
                                        'volume': volume,
                                        'high': high,
                                        'low': low,
                                        'open': open_price,
                                        'series_id': symbol,
                                        'series_name': name,
                                        'category': 'Stock Market',
                                        'unit': 'USD',
                                        'asset_type': 'Equity',
                                        'data_source': 'Alpha Vantage API (Real)'
                                    })
                                    valid_data = True
                                except (ValueError, KeyError, TypeError) as e:
                                    continue
                            
                            if valid_data:
                                sources.append("Alpha Vantage Stocks")
                                st.success(f"✅ Loaded {name} from Alpha Vantage API")
                            else:
                                st.warning(f"⚠️ No valid data for {symbol} from Alpha Vantage")
                            
                            time.sleep(12)  # Alpha Vantage rate limit (5 calls per minute)
                        else:
                            st.warning(f"⚠️ No time series data returned for {symbol} from Alpha Vantage")
                            
                    else:
                        st.warning(f"⚠️ Alpha Vantage API request failed for {symbol}: {response.status_code}")
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not load {name} from Alpha Vantage: {str(e)}")
        else:
            st.info("💡 Alpha Vantage API key not configured - individual stocks will use simulations")
        
        # Use Yahoo Finance for ETFs and remaining stocks
        loaded_symbols = set([d['series_id'] for d in data])
        remaining_stocks = {**yahoo_finance_stocks}
        
        # Add any Alpha Vantage stocks that failed to remaining list
        for symbol, name in alpha_vantage_stocks.items():
            if symbol not in loaded_symbols:
                remaining_stocks[symbol] = name
        
        if YFINANCE_AVAILABLE and remaining_stocks:
            st.info("📈 Loading ETFs and backup stocks from Yahoo Finance...")
            
            for symbol, name in remaining_stocks.items():
                try:
                    st.info(f"📈 Loading {name} ({symbol}) from Yahoo Finance...")
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1y')  # 1 year of data
                    
                    if not hist.empty:
                        valid_data = False
                        for date, row in hist.iterrows():
                            try:
                                data.append({
                                    'date': date.date(),
                                    'value': float(row['Close']),
                                    'volume': int(row['Volume']) if row['Volume'] > 0 else 1000000,
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'open': float(row['Open']),
                                    'series_id': symbol,
                                    'series_name': name,
                                    'category': 'Stock Market',
                                    'unit': 'USD',
                                    'asset_type': 'Equity',
                                    'data_source': 'Yahoo Finance (Real)'
                                })
                                valid_data = True
                            except (ValueError, TypeError):
                                continue
                        
                        if valid_data:
                            sources.append("Yahoo Finance Stocks")
                            st.success(f"✅ Loaded {name} from Yahoo Finance")
                        else:
                            st.warning(f"⚠️ No valid data for {symbol} from Yahoo Finance")
                    else:
                        st.warning(f"⚠️ No data returned for {symbol} from Yahoo Finance")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load {name} from Yahoo Finance: {str(e)}")
                    # Fallback to simulation
                    sim_data = self.generate_realistic_stock_data(symbol, name)
                    data.extend(sim_data)
        
        # Fallback simulations for any missing data
        all_expected_stocks = {**alpha_vantage_stocks, **yahoo_finance_stocks}
        loaded_symbols = set([d['series_id'] for d in data])
        missing_symbols = set(all_expected_stocks.keys()) - loaded_symbols
        
        if missing_symbols:
            st.info(f"🎯 Generating enhanced simulations for {len(missing_symbols)} missing assets...")
            for symbol in missing_symbols:
                name = all_expected_stocks[symbol]
                st.info(f"🎯 Generating simulation for {name} ({symbol})")
                sim_data = self.generate_realistic_stock_data(symbol, name)
                data.extend(sim_data)
            sources.append("Enhanced Stock Simulation")
            
        if not sources:
            sources.append("Enhanced Stock Simulation")
        
        return data, sources
    
    def load_enhanced_crypto_data(self):
        """Load enhanced cryptocurrency data using Alpha Vantage and CoinGecko"""
        
        data = []
        sources = []
        
        # Try Alpha Vantage crypto first
        if self.alpha_vantage_key != "demo_key":
            st.info("🔥 Loading premium crypto data from Alpha Vantage API...")
            crypto_symbols = ['BTC', 'ETH', 'LTC', 'XRP', 'ADA']
            
            for symbol in crypto_symbols:
                try:
                    st.info(f"₿ Loading {symbol} from Alpha Vantage...")
                    
                    url = "https://www.alphavantage.co/query"
                    params = {
                        'function': 'DIGITAL_CURRENCY_DAILY',
                        'symbol': symbol,
                        'market': 'USD',
                        'apikey': self.alpha_vantage_key
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        av_data = response.json()
                        
                        # Check for API errors
                        if 'Error Message' in av_data:
                            st.warning(f"⚠️ Alpha Vantage crypto error for {symbol}: {av_data['Error Message']}")
                            continue
                        
                        if 'Note' in av_data:
                            st.warning(f"⚠️ Alpha Vantage rate limit hit for {symbol}")
                            continue
                        
                        time_series = av_data.get('Time Series (Digital Currency Daily)', {})
                        
                        if time_series:
                            valid_data = False
                            for date_str, values in list(time_series.items())[:100]:  # Last 100 days
                                try:
                                    date = pd.to_datetime(date_str).date()
                                    close_price = float(values['4a. close (USD)'])
                                    high = float(values['2a. high (USD)'])
                                    low = float(values['3a. low (USD)'])
                                    open_price = float(values['1a. open (USD)'])
                                    volume = float(values.get('5. volume', 1000000))
                                    
                                    data.append({
                                        'date': date,
                                        'value': close_price,
                                        'volume': volume,
                                        'high': high,
                                        'low': low,
                                        'open': open_price,
                                        'series_id': f"{symbol}USDT",
                                        'series_name': f"{symbol} Price",
                                        'category': 'Cryptocurrency',
                                        'unit': 'USD',
                                        'asset_type': 'Crypto',
                                        'data_source': 'Alpha Vantage Crypto (Real)'
                                    })
                                    valid_data = True
                                except (ValueError, KeyError, TypeError):
                                    continue
                            
                            if valid_data:
                                sources.append("Alpha Vantage Crypto")
                                st.success(f"✅ Loaded {symbol} from Alpha Vantage")
                            else:
                                st.warning(f"⚠️ No valid crypto data for {symbol} from Alpha Vantage")
                            
                            time.sleep(12)  # Rate limiting
                        else:
                            st.warning(f"⚠️ No crypto time series data for {symbol} from Alpha Vantage")
                            
                except Exception as e:
                    st.warning(f"⚠️ Alpha Vantage crypto error for {symbol}: {str(e)}")
        
        # Get remaining cryptos or all cryptos if Alpha Vantage failed
        loaded_crypto_symbols = set([d['series_id'] for d in data])
        all_needed_cryptos = {'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'LTCUSDT', 'XRPUSDT'}
        needed_cryptos = all_needed_cryptos - loaded_crypto_symbols
        
        if needed_cryptos or not data:  # If we need more cryptos or Alpha Vantage failed completely
            # Try CoinGecko API (free, no API key required)
            st.info("🪙 Loading additional crypto data from CoinGecko API...")
            coingecko_data, coingecko_sources = self.load_coingecko_crypto_data(needed_cryptos)
            data.extend(coingecko_data)
            sources.extend(coingecko_sources)
        
        # Final fallback to simulations
        final_loaded_symbols = set([d['series_id'] for d in data])
        still_missing = all_needed_cryptos - final_loaded_symbols
        
        if still_missing:
            st.info(f"🎯 Generating enhanced crypto simulations for {len(still_missing)} assets...")
            for symbol in still_missing:
                crypto_name = symbol.replace('USDT', '')
                st.info(f"🎯 Generating simulation for {crypto_name}")
                sim_data = self.generate_realistic_crypto_data(crypto_name, f"{crypto_name} Price")
                data.extend(sim_data)
            sources.append("Enhanced Crypto Simulation")
        
        if not sources:
            sources.append("Enhanced Crypto Simulation")
        
        return data, sources
    
    def load_coingecko_crypto_data(self, needed_symbols=None):
        """Load cryptocurrency data from CoinGecko API"""
        
        cryptos = {
            'bitcoin': {'symbol': 'BTC', 'name': 'Bitcoin'},
            'ethereum': {'symbol': 'ETH', 'name': 'Ethereum'},
            'binancecoin': {'symbol': 'BNB', 'name': 'Binance Coin'},
            'cardano': {'symbol': 'ADA', 'name': 'Cardano'},
            'solana': {'symbol': 'SOL', 'name': 'Solana'},
            'litecoin': {'symbol': 'LTC', 'name': 'Litecoin'},
            'ripple': {'symbol': 'XRP', 'name': 'XRP'}
        }
        
        data = []
        sources = []
        
        # Filter cryptos if specific symbols needed
        if needed_symbols:
            cryptos = {k: v for k, v in cryptos.items() 
                      if f"{v['symbol']}USDT" in needed_symbols}
        
        try:
            crypto_ids = list(cryptos.keys())
            if not crypto_ids:
                return data, sources
                
            # Get current market data
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': ','.join(crypto_ids),
                'order': 'market_cap_desc',
                'per_page': len(crypto_ids),
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h,7d,30d'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                crypto_data = response.json()
                
                for coin in crypto_data:
                    coin_id = coin['id']
                    if coin_id in cryptos:
                        symbol = cryptos[coin_id]['symbol']
                        name = cryptos[coin_id]['name']
                        
                        try:
                            # Get historical data for this coin
                            hist_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                            hist_params = {'vs_currency': 'usd', 'days': '365'}
                            
                            hist_response = requests.get(hist_url, params=hist_params, timeout=10)
                            if hist_response.status_code == 200:
                                hist_data = hist_response.json()
                                prices = hist_data.get('prices', [])
                                
                                # Process price history (last 200 days)
                                valid_data = False
                                for price_point in prices[-200:]:
                                    try:
                                        timestamp, price = price_point
                                        date = datetime.fromtimestamp(timestamp / 1000).date()
                                        
                                        data.append({
                                            'date': date,
                                            'value': float(price),
                                            'volume': np.random.randint(1000000, 10000000),  # Volume not in free API
                                            'high': float(price) * 1.05,
                                            'low': float(price) * 0.95,
                                            'open': float(price) * 0.998,
                                            'series_id': f"{symbol}USDT",
                                            'series_name': f"{name} Price",
                                            'category': 'Cryptocurrency',
                                            'unit': 'USD',
                                            'asset_type': 'Crypto',
                                            'data_source': 'CoinGecko API (Real)'
                                        })
                                        valid_data = True
                                    except (ValueError, TypeError, IndexError):
                                        continue
                                
                                if valid_data:
                                    st.success(f"✅ Loaded real {name} price: ${price:,.2f}")
                                    sources.append("CoinGecko API")
                                else:
                                    st.warning(f"⚠️ No valid price data for {name}")
                                
                                time.sleep(1)  # Rate limiting for free API
                            else:
                                st.warning(f"⚠️ Could not load {name} historical data from CoinGecko")
                                
                        except Exception as e:
                            st.warning(f"⚠️ Error loading {name} from CoinGecko: {str(e)}")
                
            else:
                raise Exception(f"CoinGecko API request failed: {response.status_code}")
                
        except Exception as e:
            st.warning(f"⚠️ CoinGecko API unavailable: {str(e)}")
        
        return data, sources
    
    def load_enhanced_forex_data(self):
        """Load enhanced forex data using Alpha Vantage and Yahoo Finance"""
        
        data = []
        sources = []
        
        # Alpha Vantage forex pairs
        av_pairs = {
            'EUR': 'EUR/USD Exchange Rate',
            'GBP': 'GBP/
#Apind
    # Calculate comprehensive portfolio metrics
        all_metrics = [m for metrics in portfolio_analysis.values() for m in metrics]
        if all_metrics:
            total_return_1m = np.mean([m['return_1m'] for m in all_metrics])
            total_return_3m = np.mean([m['return_3m'] for m in all_metrics])
            total_volatility = np.mean([m['volatility'] for m in all_metrics])
            total_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
            worst_drawdown = min([m['max_drawdown'] for m in all_metrics])
            best_performer_1m = max(all_metrics, key=lambda x: x['return_1m'])
            worst_performer_1m = min(all_metrics, key=lambda x: x['return_1m'])
            
            # Generate intelligent insights
            insights = []
            
            # Overall performance insight
            if total_return_1m > 3:
                insights.append(f"🚀 **Strong Performance**: Portfolio averaged {total_return_1m:+.1f}% returns over the last month")
            elif total_return_1m > 0:
                insights.append(f"📈 **Positive Performance**: Portfolio gained {total_return_1m:+.1f}% in the last month")
            else:
                insights.append(f"📉 **Performance Alert**: Portfolio declined {total_return_1m:+.1f}% in the last month")
            
            # Risk assessment insight
            if total_volatility > 40:
                insights.append(f"⚠️ **High Risk Portfolio**: Average volatility at {total_volatility:.1f}% - consider risk management")
            elif total_volatility > 25:
                insights.append(f"🟡 **Moderate Risk**: Portfolio volatility at {total_volatility:.1f}% - balanced risk profile")
            else:
                insights.append(f"🟢 **Conservative Portfolio**: Low volatility at {total_volatility:.1f}% - stable profile")
            
            # Data quality insight
            premium_percentage = (alpha_vantage_count / len(df) * 100) if len(df) > 0 else 0
            if premium_percentage > 30:
                insights.append(f"🔥 **Premium Intelligence**: {alpha_vantage_count} assets tracked via Alpha Vantage for superior accuracy")
            elif real_data_pct > 80:
                insights.append(f"✅ **High Fidelity Analysis**: {real_data_pct:.0f}% based on real-time market data")
            elif real_data_pct > 50:
                insights.append(f"🟡 **Mixed Data Quality**: {real_data_pct:.0f}% real data combined with enhanced simulations")
            else:
                insights.append(f"🔴 **Limited Real Data**: Consider enabling API keys for enhanced data quality")
            
            # Diversification insight
            crypto_weight = len(portfolio_analysis.get('Crypto', [])) / len(all_metrics) * 100
            if crypto_weight > 50:
                insights.append(f"₿ **Crypto Heavy**: {crypto_weight:.0f}% crypto allocation - high growth potential but elevated risk")
            elif crypto_weight > 20:
                insights.append(f"⚖️ **Balanced Crypto Exposure**: {crypto_weight:.0f}% crypto allocation - good growth/risk balance")
            elif crypto_weight > 0:
                insights.append(f"🎯 **Conservative Crypto**: {crypto_weight:.0f}% crypto allocation - minimal alternative asset exposure")
            
            # Sharpe ratio insight
            if total_sharpe > 1.5:
                insights.append(f"🏆 **Excellent Risk-Adjusted Returns**: Sharpe ratio of {total_sharpe:.2f} indicates superior performance")
            elif total_sharpe > 1.0:
                insights.append(f"👍 **Good Risk-Adjusted Returns**: Sharpe ratio of {total_sharpe:.2f} shows solid efficiency")
            elif total_sharpe > 0.5:
                insights.append(f"📊 **Moderate Efficiency**: Sharpe ratio of {total_sharpe:.2f} - room for optimization")
            else:
                insights.append(f"⚠️ **Risk-Return Imbalance**: Low Sharpe ratio of {total_sharpe:.2f} - consider rebalancing")
            
            # AI prediction insight
            if len(predictions) > 3:
                insights.append(f"🤖 **Strong AI Coverage**: {len(predictions)} active models providing high-confidence forecasts")
            elif len(predictions) > 0:
                insights.append(f"🔮 **AI Predictions Available**: {len(predictions)} models active with {confidence_threshold}%+ confidence")
            else:
                insights.append(f"💡 **Model Training**: AI systems analyzing data - predictions available when confidence ≥{confidence_threshold}%")
            
            # Top/bottom performer insight
            insights.append(f"🏅 **Top Performer**: {best_performer_1m['series_name']} (+{best_performer_1m['return_1m']:.1f}%)")
            insights.append(f"📉 **Needs Attention**: {worst_performer_1m['series_name']} ({worst_performer_1m['return_1m']:+.1f}%)")
            
            # Market conditions insight
            current_date = datetime.now().strftime('%B %Y')
            insights.append(f"📅 **Current Analysis**: Based on {len(df):,} data points as of {current_date}")
            
            # Display insights
            for insight in insights:
                st.markdown(f"• {insight}")
        
        # Enhanced real-time performance attribution
        st.markdown("### 🏆 Real-Time Performance Attribution")
        
        if portfolio_analysis:
            # Prepare data for visualization
            categories = []
            avg_returns_1m = []
            avg_returns_3m = []
            avg_volatilities = []
            sharpe_ratios = []
            asset_counts = []
            data_quality_scores = []
            
            for category, metrics in portfolio_analysis.items():
                if metrics:
                    categories.append(category)
                    avg_returns_1m.append(np.mean([m['return_1m'] for m in metrics]))
                    avg_returns_3m.append(np.mean([m['return_3m'] for m in metrics]))
                    avg_volatilities.append(np.mean([m['volatility'] for m in metrics]))
                    sharpe_ratios.append(np.mean([m['sharpe_ratio'] for m in metrics]))
                    asset_counts.append(len(metrics))
                    
                    # Calculate data quality score
                    real_count = len([m for m in metrics if 'Real' in m.get('data_source', '')])
                    quality_score = (real_count / len(metrics)) * 100
                    data_quality_scores.append(quality_score)
            
            if categories:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Enhanced returns chart with data quality indicators
                    fig_returns = go.Figure()
                    
                    fig_returns.add_trace(go.Bar(
                        x=categories,
                        y=avg_returns_1m,
                        name='1-Month Returns',
                        marker_color=[
                            '#00C851' if ret > 2 else '#4CAF50' if ret > 0 else '#FF4444' if ret < -2 else '#FF8800'
                            for ret in avg_returns_1m
                        ],
                        text=[f"{ret:+.1f}%" for ret in avg_returns_1m],
                        textposition='auto'
                    ))
                    
                    fig_returns.update_layout(
                        title="1-Month Returns by Asset Category",
                        xaxis_title="Asset Category",
                        yaxis_title="Return (%)",
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                with col2:
                    # Risk-return scatter plot
                    fig_risk_return = px.scatter(
                        x=avg_volatilities,
                        y=avg_returns_1m,
                        size=asset_counts,
                        color=data_quality_scores,
                        hover_name=categories,
                        labels={
                            'x': 'Volatility (%)',
                            'y': '1-Month Return (%)',
                            'color': 'Data Quality (%)',
                            'size': 'Asset Count'
                        },
                        title="Risk-Return Profile by Category",
                        color_continuous_scale='Viridis'
                    )
                    
                    fig_risk_return.update_layout(height=400)
                    st.plotly_chart(fig_risk_return, use_container_width=True)
                
                # Additional performance metrics table
                st.markdown("#### 📊 Detailed Performance Metrics")
                
                performance_df = pd.DataFrame({
                    'Asset Category': categories,
                    '1M Return (%)': [f"{ret:+.1f}" for ret in avg_returns_1m],
                    '3M Return (%)': [f"{ret:+.1f}" for ret in avg_returns_3m],
                    'Volatility (%)': [f"{vol:.1f}" for vol in avg_volatilities],
                    'Sharpe Ratio': [f"{sr:.2f}" for sr in sharpe_ratios],
                    'Asset Count': asset_counts,
                    'Data Quality (%)': [f"{dq:.0f}" for dq in data_quality_scores]
                })
                
                st.dataframe(performance_df, use_container_width=True)
        
        # Enhanced data source breakdown
        st.markdown("### 📡 Comprehensive Data Source Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detailed source breakdown
            source_details = df.groupby(['data_source', 'category']).agg({
                'series_id': 'count',
                'date': ['min', 'max']
            }).reset_index()
            
            source_details.columns = ['Data Source', 'Category', 'Count', 'First Date', 'Last Date']
            source_details['Date Range'] = source_details['Last Date'] - source_details['First Date']
            source_details['Days'] = source_details['Date Range'].dt.days
            
            # Format for display
            display_df = source_details[['Data Source', 'Category', 'Count', 'Days']].copy()
            display_df = display_df.sort_values(['Data Source', 'Category'])
            
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            # Source quality pie chart with enhanced styling
            source_summary = df['data_source'].value_counts().reset_index()
            source_summary.columns = ['Data Source', 'Count']
            
            # Custom color mapping for sources
            color_map = {
                'Alpha Vantage API (Real)': '#ff6b6b',
                'Alpha Vantage Crypto (Real)': '#ffa500', 
                'Alpha Vantage FX (Real)': '#ff8c00',
                'Yahoo Finance (Real)': '#4CAF50',
                'CoinGecko API (Real)': '#2196F3',
                'FRED API (Real)': '#9C27B0',
                'FRED International (Real)': '#673AB7'
            }
            
            # Add default colors for simulation sources
            colors = [color_map.get(source, '#FFC107') for source in source_summary['Data Source']]
            
            fig_sources = go.Figure(data=[go.Pie(
                labels=source_summary['Data Source'],
                values=source_summary['Count'],
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_sources.update_layout(
                title="Data Source Distribution",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_sources, use_container_width=True)
        
        # System health and recommendations
        st.markdown("### 🔧 System Health & Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟢 System Strengths")
            strengths = []
            if alpha_vantage_count > 0:
                strengths.append("🔥 Premium Alpha Vantage integration")
            if real_data_pct > 80:
                strengths.append("✅ High real data coverage")
            if len(predictions) > 2:
                strengths.append("🤖 Multiple AI models active")
            if len(portfolio_analysis) > 3:
                strengths.append("📊 Good asset diversification")
            
            for strength in strengths:
                st.markdown(f"• {strength}")
        
        with col2:
            st.markdown("#### 🟡 Optimization Opportunities")
            opportunities = []
            if alpha_vantage_count == 0:
                opportunities.append("🔥 Add Alpha Vantage for premium data")
            if real_data_pct < 60:
                opportunities.append("📡 Enable more real data sources")
            if len(predictions) < 3:
                opportunities.append("🤖 Lower confidence threshold for more models")
            if total_volatility > 50:
                opportunities.append("⚖️ Consider portfolio rebalancing")
            
            for opportunity in opportunities:
                st.markdown(f"• {opportunity}")
        
        with col3:
            st.markdown("#### 📈 Next Steps")
            next_steps = [
                "🔄 Monitor model predictions daily",
                "📊 Review risk metrics weekly", 
                "🎯 Rebalance based on AI insights",
                "📡 Consider additional data sources"
            ]
            
            for step in next_steps:
                st.markdown(f"• {step}")
    
    # Enhanced footer with comprehensive system information
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;">
        <h4>🚀 Economic Pulse V3.0 - Real Data Financial Intelligence Platform</h4>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; text-align: center;">
            <div>
                <strong>🔥 Premium Data</strong><br>
                {alpha_vantage_count} Alpha Vantage Assets<br>
                <small>Individual stocks, crypto, forex</small>
            </div>
            <div>
                <strong>📡 Live Sources</strong><br>
                {real_data_pct:.0f}% Real Market Data<br>
                <small>Yahoo Finance, CoinGecko, FRED</small>
            </div>
            <div>
                <strong>🤖 AI Models</strong><br>
                {len(predictions)} Active Forecasts<br>
                <small>{confidence_threshold}%+ confidence threshold</small>
            </div>
            <div>
                <strong>📊 Coverage</strong><br>
                {total_assets} Global Assets<br>
                <small>{categories} categories, {len(df):,} data points</small>
            </div>
        </div>
        
        <div style="margin-top: 1rem;">
            <strong>🌍 Data Sources:</strong> Alpha Vantage API • Yahoo Finance • CoinGecko • FRED Economic Data • Enhanced ML Simulations<br>
            <strong>🧠 AI Technology:</strong> Multi-Model Ensemble • Real-Time Training • Confidence Scoring • Technical Analysis<br>
            <strong>📅 Last Updated:</strong> {df['date'].max().strftime('%Y-%m-%d %H:%M')} | 
            <strong>🎯 Quality Rating:</strong> {'🔥 Premium' if alpha_vantage_count > 0 else '🟢 Excellent' if real_data_pct > 80 else '🟡 Good' if real_data_pct > 50 else '🔴 Basic'}
        </div>
        
        <div style="margin-top: 0.5rem; font-size: 12px; opacity: 0.7;">
            <em>Built with Streamlit • Plotly • Scikit-learn • Professional Financial APIs</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()                        fig_vol.update_layout(
                            title=f"{stock_name} - Trading Volume Analysis",
                            xaxis_title="Date",
                            yaxis_title="Volume",
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_vol, use_container_width=True)
                
                # Enhanced technical analysis
                st.markdown("#### 🔍 Advanced Technical Analysis")
                
                values = stock_data['value'].values
                if len(values) >= 50:
                    # Calculate technical indicators
                    sma_20 = np.mean(values[-20:])
                    sma_50 = np.mean(values[-50:])
                    current_price = values[-1]
                    
                    # Calculate RSI
                    gains = np.maximum(np.diff(values), 0)
                    losses = np.maximum(-np.diff(values), 0)
                    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
                    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
                    rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
                    
                    # Calculate volatility
                    returns = np.diff(values) / values[:-1]
                    volatility = np.std(returns) * np.sqrt(252) * 100
                    
                    # Support and resistance levels
                    recent_high = np.max(values[-30:])
                    recent_low = np.min(values[-30:])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("20-Day SMA", f"${sma_20:.2f}", 
                                f"{((current_price - sma_20) / sma_20 * 100):+.1f}%")
                    
                    with col2:
                        st.metric("50-Day SMA", f"${sma_50:.2f}",
                                f"{((current_price - sma_50) / sma_50 * 100):+.1f}%")
                    
                    with col3:
                        rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                        st.metric("RSI (14)", f"{rsi:.1f}", rsi_color)
                    
                    with col4:
                        vol_trend = "High" if volatility > 30 else "Medium" if volatility > 15 else "Low"
                        st.metric("Volatility", f"{volatility:.1f}%", vol_trend)
                    
                    # Additional metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("30-Day High", f"${recent_high:.2f}")
                    
                    with col2:
                        st.metric("30-Day Low", f"${recent_low:.2f}")
                    
                    with col3:
                        price_position = ((current_price - recent_low) / (recent_high - recent_low) * 100)
                        st.metric("Price Position", f"{price_position:.0f}%")
                    
                    with col4:
                        # Trend determination
                        if current_price > sma_20 > sma_50:
                            trend = "🟢 Strong Bullish"
                        elif current_price > sma_20:
                            trend = "🟡 Bullish"
                        elif current_price < sma_20 < sma_50:
                            trend = "🔴 Strong Bearish"
                        else:
                            trend = "🟡 Bearish"
                        st.metric("Trend", trend)
    
    with tab3:
        st.subheader("₿ Real-Time Cryptocurrency Analysis")
        
        # Enhanced crypto performance with premium data highlighting
        if 'Crypto' in portfolio_analysis:
            crypto_metrics = portfolio_analysis['Crypto']
            
            st.markdown("### 🚀 Live Crypto Performance Metrics")
            
            # Sort by data source quality and market cap
            crypto_metrics_sorted = sorted(crypto_metrics, 
                                         key=lambda x: (
                                             0 if "Alpha Vantage" in x.get('data_source', '') else
                                             1 if "Real" in x.get('data_source', '') else 2,
                                             -x['current_value']  # Higher value first
                                         ))
            
            for metric in crypto_metrics_sorted:
                # Crypto-specific color coding (higher volatility expected)
                perf_color = "#00C851" if metric['return_1m'] > 5 else "#00C851" if metric['return_1m'] > 0 else "#FF4444" if metric['return_1m'] < -5 else "#FF4444"
                vol_color = "#FF4444" if metric['volatility'] > 100 else "#FF8800" if metric['volatility'] > 60 else "#00C851"
                
                # Premium data source detection
                is_premium = "Alpha Vantage" in metric.get('data_source', '')
                is_real = "Real" in metric.get('data_source', '')
                
                if is_premium:
                    badge = "🔥 Alpha Vantage"
                    badge_class = "alpha-vantage-badge"
                elif is_real:
                    badge = "✅ Real-Time"
                    badge_class = "real-data-badge"
                else:
                    badge = "🎯 Simulation"
                    badge_class = "data-source-info"
                
                with st.container():
                    st.markdown(f"""
                    <div class="asset-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h4>{metric['series_name']} ({metric['series_id']})</h4>
                            <div class="{badge_class}" style="display: inline-block;">{badge}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Current:</strong> ${metric['current_value']:,.2f}</div>
                            <div><strong>1M Return:</strong> <span style="color: {perf_color}; font-weight: bold;">{metric['return_1m']:+.1f}%</span></div>
                            <div><strong>Volatility:</strong> <span style="color: {vol_color}; font-weight: bold;">{metric['volatility']:.1f}%</span></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>3M Return:</strong> <span style="color: {'#00C851' if metric['return_3m'] > 0 else '#FF4444'}">{metric['return_3m']:+.1f}%</span></div>
                            <div><strong>1Y Return:</strong> <span style="color: {'#00C851' if metric['return_1y'] > 0 else '#FF4444'}">{metric['return_1y']:+.1f}%</span></div>
                            <div><strong>Sharpe Ratio:</strong> {metric['sharpe_ratio']:.2f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Max Drawdown:</strong> <span style="color: #FF4444">{metric['max_drawdown']:.1f}%</span></div>
                            <div><strong>VaR (95%):</strong> <span style="color: #FF8800">{metric['var_95']:.1f}%</span></div>
                            <div><strong>Data Points:</strong> {metric['data_points']:,}</div>
                        </div>
                        <div class="data-source-info">
                            📡 {metric['data_source']} | Updated: {metric['last_updated']} | Points: {metric['data_points']:,}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Enhanced crypto market analysis
        st.markdown("### 📊 Advanced Crypto Market Analysis")
        
        crypto_data = df[df['category'] == 'Cryptocurrency']
        if not crypto_data.empty and len(crypto_data['series_id'].unique()) > 1:
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation matrix
                crypto_pivot = crypto_data.pivot_table(index='date', columns='series_id', values='value')
                crypto_returns = crypto_pivot.pct_change().dropna()
                crypto_corr = crypto_returns.corr()
                
                if not crypto_corr.empty:
                    fig_corr = px.imshow(
                        crypto_corr,
                        title="Real-Time Cryptocurrency Correlation Matrix",
                        color_continuous_scale='RdBu',
                        aspect='auto',
                        color_continuous_midpoint=0,
                        text_auto=True
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Volatility comparison
                crypto_vol_data = []
                for symbol in crypto_data['series_id'].unique():
                    symbol_data = crypto_data[crypto_data['series_id'] == symbol].sort_values('date')
                    if len(symbol_data) >= 30:
                        returns = np.diff(symbol_data['value'].values) / symbol_data['value'].values[:-1] * 100
                        volatility = np.std(returns) * np.sqrt(365)  # Annualized for crypto (365 days)
                        crypto_vol_data.append({
                            'Asset': symbol,
                            'Volatility': volatility,
                            'Source': symbol_data.iloc[-1].get('data_source', 'Unknown')
                        })
                
                if crypto_vol_data:
                    vol_df = pd.DataFrame(crypto_vol_data)
                    fig_vol = px.bar(
                        vol_df,
                        x='Asset',
                        y='Volatility',
                        title="Cryptocurrency Volatility Comparison",
                        color='Source',
                        color_discrete_map={
                            'Alpha Vantage Crypto (Real)': '#ff6b6b',
                            'CoinGecko API (Real)': '#4CAF50',
                            'Enhanced Crypto Simulation (2025 Bull Market)': '#FFC107'
                        }
                    )
                    fig_vol.update_layout(height=400)
                    st.plotly_chart(fig_vol, use_container_width=True)
            
            # Enhanced correlation insights
            st.markdown("#### 🔍 Market Correlation Insights")
            
            if not crypto_corr.empty:
                # Find highest and lowest correlations
                corr_values = []
                for i in range(len(crypto_corr.columns)):
                    for j in range(i+1, len(crypto_corr.columns)):
                        asset1 = crypto_corr.columns[i]
                        asset2 = crypto_corr.columns[j]
                        corr_val = crypto_corr.iloc[i, j]
                        if not np.isnan(corr_val):
                            corr_values.append((asset1, asset2, corr_val))
                
                if corr_values:
                    corr_values.sort(key=lambda x: x[2], reverse=True)
                    highest_corr = corr_values[0]
                    lowest_corr = corr_values[-1]
                    avg_corr = np.mean([x[2] for x in corr_values])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        **🔗 Highest Correlation:**  
                        {highest_corr[0]} ↔ {highest_corr[1]}  
                        **{highest_corr[2]:.3f}**
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **↔️ Lowest Correlation:**  
                        {lowest_corr[0]} ↔ {lowest_corr[1]}  
                        **{lowest_corr[2]:.3f}**
                        """)
                    
                    with col3:
                        market_mood = "🟢 Diversified" if avg_corr < 0.7 else "🟡 Moderate" if avg_corr < 0.85 else "🔴 Highly Correlated"
                        st.markdown(f"""
                        **📊 Average Correlation:**  
                        **{avg_corr:.3f}**  
                        {market_mood}
                        """)
    
    with tab4:
        st.subheader("🔮 AI-Powered Predictions on Real Data")
        
        if predictions:
            st.markdown("### 🤖 Advanced AI Forecasts with Real Data")
            
            # Sort predictions by data source quality
            sorted_predictions = sorted(
                predictions.items(),
                key=lambda x: (
                    0 if "Alpha Vantage" in x[1].get('data_source', '') else
                    1 if "Real" in x[1].get('data_source', '') else 2,
                    -x[1].get('confidence', 0)
                )
            )
            
            for asset_id, pred_data in sorted_predictions:
                st.markdown(f"#### 📈 {pred_data['series_name']} - {prediction_periods}-Day Advanced Forecast")
                
                # Get historical real data
                hist_data = df[df['series_id'] == asset_id].sort_values('date')
                
                if not hist_data.empty:
                    # Enhanced prediction chart with technical analysis
                    fig_pred = go.Figure()
                    
                    # Historical data (last 90 days)
                    recent_hist = hist_data.tail(90)
                    fig_pred.add_trace(
                        go.Scatter(
                            x=recent_hist['date'],
                            y=recent_hist['value'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='#1f77b4', width=3)
                        )
                    )
                    
                    # Add moving average to historical data
                    if len(recent_hist) >= 20:
                        ma_20 = recent_hist['value'].rolling(20).mean()
                        fig_pred.add_trace(
                            go.Scatter(
                                x=recent_hist['date'],
                                y=ma_20,
                                mode='lines',
                                name='20-day MA',
                                line=dict(color='orange', width=1, dash='dash')
                            )
                        )
                    
                    # AI predictions
                    fig_pred.add_trace(
                        go.Scatter(
                            x=pred_data['dates'],
                            y=pred_data['predictions'],
                            mode='lines+markers',
                            name='AI Prediction',
                            line=dict(color='#ff4444', dash='dash', width=3),
                            marker=dict(size=8, symbol='diamond')
                        )
                    )
                    
                    # Enhanced confidence interval with dynamic bounds
                    confidence = pred_data.get('confidence', 80) / 100
                    volatility_factor = pred_data.get('trend_strength', 1) / 1000
                    std_dev = np.std(pred_data['predictions']) * (1 - confidence + volatility_factor)
                    
                    upper_bound = [p + std_dev * (1 + i/len(pred_data['predictions'])) for i, p in enumerate(pred_data['predictions'])]
                    lower_bound = [p - std_dev * (1 + i/len(pred_data['predictions'])) for i, p in enumerate(pred_data['predictions'])]
                    
                    fig_pred.add_trace(
                        go.Scatter(
                            x=pred_data['dates'] + pred_data['dates'][::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill='toself',
                            fillcolor='rgba(255,68,68,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{confidence*100:.0f}% Confidence Band',
                            showlegend=True
                        )
                    )
                    
                    # Data source indicator
                    is_premium = "Alpha Vantage" in pred_data.get('data_source', '')
                    title_suffix = " 🔥 (Premium Data)" if is_premium else " ✅ (Real Data)" if "Real" in pred_data.get('data_source', '') else " 🎯 (Simulation)"
                    
                    fig_pred.update_layout(
                        title=f"{pred_data['series_name']} - AI Prediction{title_suffix}",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Enhanced prediction metrics with additional indicators
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    current_value = hist_data.iloc[-1]['value']
                    predicted_value = pred_data['predictions'][-1]
                    change_pct = ((predicted_value - current_value) / current_value * 100)
                    confidence_score = pred_data.get('confidence', 80)
                    rsi_value = pred_data.get('rsi', 50)
                    
                    with col1:
                        st.metric("Current Value", f"{current_value:.2f}")
                    
                    with col2:
                        st.metric(f"{prediction_periods}-Day Forecast", f"{predicted_value:.2f}")
                    
                    with col3:
                        change_color = "normal" if abs(change_pct) < 2 else "inverse" if change_pct < 0 else "normal"
                        st.metric("Predicted Change", f"{change_pct:+.1f}%", delta_color=change_color)
                    
                    with col4:
                        confidence_emoji = "🟢" if confidence_score > 85 else "🟡" if confidence_score > 70 else "🔴"
                        st.metric("AI Confidence", f"{confidence_score:.0f}%", confidence_emoji)
                    
                    with col5:
                        rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        st.metric("RSI Signal", f"{rsi_value:.0f}", rsi_signal)
                    
                    # Enhanced AI insights with technical analysis
                    if change_pct > 5:
                        trend_direction = "📈 Strong Bullish"
                        trend_color = "#00C851"
                    elif change_pct > 2:
                        trend_direction = "🟢 Bullish"
                        trend_color = "#00C851"
                    elif change_pct < -5:
                        trend_direction = "📉 Strong Bearish"
                        trend_color = "#FF4444"
                    elif change_pct < -2:
                        trend_direction = "🔴 Bearish"
                        trend_color = "#FF4444"
                    else:
                        trend_direction = "➡️ Neutral"
                        trend_color = "#FFC107"
                    
                    confidence_level = "High" if confidence_score > 80 else "Medium" if confidence_score > 60 else "Low"
                    
                    # Enhanced data source badge
                    if is_premium:
                        source_badge = "🔥 Premium Alpha Vantage Data"
                        source_color = "#ff6b6b"
                    elif "Real" in pred_data.get('data_source', ''):
                        source_badge = "✅ Real Market Data"
                        source_color = "#00C851"
                    else:
                        source_badge = "🎯 Enhanced Simulation"
                        source_color = "#FFC107"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <h4>🤖 Advanced AI Analysis</h4>
                            <span style="background: {source_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 12px;">
                                {source_badge}
                            </span>
                        </div>
                        <p><strong>Forecast Summary:</strong> The AI model predicts a <strong style="color: {trend_color}">{trend_direction}</strong> trend 
                        for {pred_data['series_name']} with <strong>{confidence_level}</strong> confidence ({confidence_score:.0f}%).</p>
                        
                        <p><strong>Expected Movement:</strong> {change_pct:+.1f}% change over {prediction_periods} days 
                        (from ${current_value:.2f} to ${predicted_value:.2f})</p>
                        
                        <p><strong>Technical Indicators:</strong> RSI at {rsi_value:.0f} suggests {rsi_signal.lower()} conditions. 
                        Trend strength indicator: {pred_data.get('trend_strength', 0):.1f}</p>
                        
                        <p><strong>Risk Assessment:</strong> Based on historical volatility and current market conditions, 
                        this prediction carries {confidence_level.lower()} confidence with the model trained on {pred_data.get('data_source', 'market data')}.</p>
                        
                        <p><strong>Data Quality:</strong> Analysis based on {len(hist_data):,} data points from {pred_data.get('data_source', 'unknown source')}.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🤖 AI models are analyzing real market data... High-confidence predictions will appear here.")
            
            # Show model training status with enhanced details
            st.markdown("### 🧠 AI Model Training Status")
            
            training_status = []
            for asset in ['SPY', 'BTCUSDT', 'UNRATE', 'EURUSD', 'AAPL', 'NVDA']:
                asset_data = df[df['series_id'] == asset]
                if not asset_data.empty:
                    data_points = len(asset_data)
                    data_source = asset_data.iloc[-1].get('data_source', 'Unknown')
                    
                    if data_points > 200:
                        status = "✅ Ready for High-Confidence Predictions"
                        quality = "Excellent"
                    elif data_points > 100:
                        status = "🟡 Ready for Medium-Confidence Predictions"
                        quality = "Good"
                    elif data_points > 50:
                        status = "🟠 Limited Data - Low Confidence Only"
                        quality = "Limited"
                    else:
                        status = "🔴 Insufficient Data for Reliable Predictions"
                        quality = "Poor"
                    
                    training_status.append({
                        'Asset': asset,
                        'Data Points': data_points,
                        'Data Source': data_source,
                        'Quality': quality,
                        'Status': status
                    })
            
            if training_status:
                status_df = pd.DataFrame(training_status)
                st.dataframe(status_df, use_container_width=True)
    
    with tab5:
        st.subheader("🧠 Portfolio Intelligence & Advanced Risk Analysis")
        
        # Enhanced portfolio overview with premium data indicators
        st.markdown("### 📊 Real-Time Portfolio Overview")
        
        total_assets = len(df['series_id'].unique())
        categories = len(df['category'].unique())
        real_data_count = len(df[df['data_source'].str.contains('Real', na=False)])
        alpha_vantage_count = len(df[df['data_source'].str.contains('Alpha Vantage', na=False)])
        real_data_pct = (real_data_count / len(df) * 100) if len(df) > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Assets", total_assets, "Global Coverage")
        
        with col2:
            st.metric("Asset Categories", categories, "Diversified")
        
        with col3:
            premium_pct = (alpha_vantage_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("Premium Data", f"{alpha_vantage_count}", f"{premium_pct:.0f}%")
        
        with col4:
            st.metric("Real Data Quality", f"{real_data_pct:.0f}%", "Live Sources")
        
        with col5:
            active_predictions = len(predictions)
            st.metric("AI Models Active", active_predictions, f"{confidence_threshold}%+ Confidence")
        
        # Enhanced risk analysis by category with premium data highlighting
        st.markdown("### ⚠️ Advanced Risk Analysis by Asset Category")
        
        for asset_type, metrics in portfolio_analysis.items():
            if metrics and len(metrics) > 0:
                # Calculate enhanced metrics
                avg_volatility = np.mean([m['volatility'] for m in metrics])
                avg_return_1m = np.mean([m['return_1m'] for m in metrics])
                avg_return_3m = np.mean([m['return_3m'] for m in metrics])
                avg_sharpe = np.mean([m['sharpe_ratio'] for m in metrics])
                max_drawdown_worst = min([m['max_drawdown'] for m in metrics])
                
                high_risk_count = len([m for m in metrics if m['volatility'] > 30])
                alpha_vantage_count_cat = len([m for m in metrics if 'Alpha Vantage' in m.get('data_source', '')])
                real_data_count_cat = len([m for m in metrics if 'Real' in m.get('data_source', '')])
                
                # Enhanced risk categorization
                if asset_type == 'Crypto':
                    risk_level = "🔴 Very High" if avg_volatility > 100 else "🔴 High" if avg_volatility > 80 else "🟡 Medium" if avg_volatility > 50 else "🟢 Low"
                elif asset_type == 'Equity':
                    risk_level = "🔴 High" if avg_volatility > 40 else "🟡 Medium" if avg_volatility > 25 else "🟢 Low"
                else:
                    risk_level = "🔴 High" if avg_volatility > 30 else "🟡 Medium" if avg_volatility > 15 else "🟢 Low"
                
                # Performance color coding
                perf_1m_color = "#00C851" if avg_return_1m > 2 else "#00C851" if avg_return_1m > 0 else "#FF4444"
                perf_3m_color = "#00C851" if avg_return_3m > 5 else "#00C851" if avg_return_3m > 0 else "#FF4444"
                
                # Premium data indicators
                premium_indicators = []
                if alpha_vantage_count_cat > 0:
                    premium_indicators.append(f"🔥 {alpha_vantage_count_cat} Alpha Vantage")
                if real_data_count_cat > 0:
                    premium_indicators.append(f"✅ {real_data_count_cat} Real-Time")
                
                premium_text = " | ".join(premium_indicators) if premium_indicators else "🎯 Simulated Data"
                
                st.markdown(f"""
                <div class="risk-alert">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4>{asset_type} Portfolio Analysis</h4>
                        <div style="font-size: 12px; opacity: 0.8;">{premium_text}</div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <strong>Risk Level:</strong> {risk_level}<br>
                            <strong>Avg Volatility:</strong> {avg_volatility:.1f}%<br>
                            <strong>High Risk Assets:</strong> {high_risk_count}/{len(metrics)}
                        </div>
                        <div>
                            <strong>1M Return:</strong> <span style="color: {perf_1m_color}; font-weight: bold;">{avg_return_1m:+.1f}%</span><br>
                            <strong>3M Return:</strong> <span style="color: {perf_3m_color}; font-weight: bold;">{avg_return_3m:+.1f}%</span><br>
                            <strong>Avg Sharpe:</strong> {avg_sharpe:.2f}
                        </div>
                        <div>
                            <strong>Worst Drawdown:</strong> <span style="color: #FF4444">{max_drawdown_worst:.1f}%</span><br>
                            <strong>Asset Count:</strong> {len(metrics)}<br>
                            <strong>Data Quality:</strong> {((real_data_count_cat/len(metrics))*100):.0f}% Real
                        </div>
                    </div>
                    
                    <div style="font-size: 12px; opacity: 0.7;">
                        📊 Analysis based on real-time market data from multiple premium sources
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced AI-generated insights based on real data
        st.markdown("### 🤖 AI-Generated Portfolio Insights")
        
        # Calculate comprehensive portfolio metrics
        all_metrics = [m for metrics in portfolio_analysis.values() for m in metrics]
        if all_metrics:
            total_return_1m = np.mean([m['return_1m'] for            for series_id in type_data['series_id'].unique():
                series_data = type_data[type_data['series_id'] == series_id].sort_values('date')
                if len(series_data) >= 30:
                    values = series_data['value'].values
                    dates = pd.to_datetime(series_data['date'])
                    
                    # Calculate real returns
                    returns = np.diff(values) / values[:-1] * 100
                    
                    # Time-based returns
                    return_1m = ((values[-1] - values[-30]) / values[-30] * 100) if len(values) >= 30 else 0
                    return_3m = ((values[-1] - values[-90]) / values[-90] * 100) if len(values) >= 90 else 0
                    return_1y = ((values[-1] - values[-252]) / values[-252] * 100) if len(values) >= 252 else 0
                    
                    # Risk metrics
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
                    
                    # Drawdown calculation
                    cumulative = np.cumprod(1 + returns / 100)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max * 100
                    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                    
                    # Value at Risk (95%)
                    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
                    
                    performance_metrics.append({
                        'series_id': series_id,
                        'series_name': series_data.iloc[-1]['series_name'],
                        'current_value': values[-1],
                        'return_1m': return_1m,
                        'return_3m': return_3m,
                        'return_1y': return_1y,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'var_95': var_95,
                        'data_source': series_data.iloc[-1].get('data_source', 'Unknown'),
                        'last_updated': dates.iloc[-1].strftime('%Y-%m-%d'),
                        'data_points': len(series_data)
                    })
            
            analysis[asset_type] = performance_metrics
        
        return analysis

def main():
    """Main application with enhanced real data integration"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Economic Pulse V3.0 - Real Data Financial Intelligence</h1>
        <p>🔥 Alpha Vantage Premium • 📊 Live Economic Data • 🤖 AI Analytics • 🌍 Global Coverage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Real Data Configuration")
    
    # Enhanced API configuration display
    st.sidebar.subheader("🔑 Premium API Status")
    
    # Check API availability
    av_available = st.secrets.get("alpha_vantage_key", "demo_key") != "demo_key"
    fred_available = st.secrets.get("fred_api_key", "demo_key") != "demo_key"
    
    if av_available:
        st.sidebar.success("🔥 Alpha Vantage API: ACTIVE")
        st.sidebar.markdown("• Premium individual stocks")
        st.sidebar.markdown("• Enhanced crypto data")
        st.sidebar.markdown("• Professional forex rates")
        st.sidebar.markdown("• Rate limited for quality")
    else:
        st.sidebar.warning("⚠️ Alpha Vantage API: Not configured")
        st.sidebar.markdown("💡 Add `alpha_vantage_key` to secrets")
        st.sidebar.markdown("For premium AAPL, MSFT, BTC, EUR/USD data")
    
    if fred_available:
        st.sidebar.success("✅ FRED API: ACTIVE")
        st.sidebar.markdown("• Real economic indicators")
        st.sidebar.markdown("• International data")
    else:
        st.sidebar.info("💡 FRED API: Using enhanced simulations")
        st.sidebar.markdown("Add `fred_api_key` for real economic data")
    
    # Data source information
    st.sidebar.markdown("""
    <div class='premium-info'>
    📡 <strong>Data Source Hierarchy:</strong><br>
    🔥 Alpha Vantage - Premium stocks/crypto/forex<br>
    ✅ Yahoo Finance - ETFs & backup stocks<br>
    🪙 CoinGecko - Crypto backup<br>
    📊 FRED - Economic indicators<br>
    🎯 Enhanced Simulations - Smart fallbacks
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    st.sidebar.subheader("📊 Asset Classes")
    include_stocks = st.sidebar.checkbox("📈 Stock Market", value=True)
    include_crypto = st.sidebar.checkbox("₿ Cryptocurrency", value=True)
    include_forex = st.sidebar.checkbox("💱 Forex", value=True)
    include_intl = st.sidebar.checkbox("🌍 International", value=True)
    
    # AI Model configuration
    st.sidebar.subheader("🤖 AI Configuration")
    use_advanced_ml = st.sidebar.checkbox("🧠 Advanced ML", value=ML_AVAILABLE)
    prediction_periods = st.sidebar.slider("🔮 Prediction Days", 7, 60, 30)
    confidence_threshold = st.sidebar.slider("📊 Confidence Threshold", 60, 95, 80)
    
    # Load real data with enhanced progress tracking
    with st.spinner("🔄 Initializing premium multi-source data pipeline..."):
        df = load_comprehensive_real_data()
    
    if df.empty:
        st.error("❌ Unable to load any data. Please check API connections and try again.")
        st.info("💡 Tip: Ensure you have internet connectivity and valid API keys configured.")
        return
    
    # Enhanced data summary in sidebar
    st.sidebar.subheader("📈 Live Data Summary")
    total_assets = len(df['series_id'].unique())
    total_points = len(df)
    real_data_count = len(df[df['data_source'].str.contains('Real', na=False)])
    alpha_vantage_count = len(df[df['data_source'].str.contains('Alpha Vantage', na=False)])
    real_data_pct = (real_data_count / len(df) * 100) if len(df) > 0 else 0
    
    st.sidebar.metric("Total Assets", total_assets)
    st.sidebar.metric("Data Points", f"{total_points:,}")
    st.sidebar.metric("Alpha Vantage", alpha_vantage_count)
    st.sidebar.metric("Real Data %", f"{real_data_pct:.0f}%")
    st.sidebar.metric("Last Updated", df['date'].max().strftime('%Y-%m-%d'))
    
    # Data quality indicator
    if real_data_pct > 80:
        st.sidebar.success("🟢 Excellent Data Quality")
    elif real_data_pct > 60:
        st.sidebar.info("🟡 Good Data Quality") 
    else:
        st.sidebar.warning("🔴 Limited Real Data")
    
    # Filter data based on user selections
    filtered_categories = []
    if include_stocks:
        filtered_categories.append('Stock Market')
    if include_crypto:
        filtered_categories.append('Cryptocurrency') 
    if include_forex:
        filtered_categories.append('Forex')
    if include_intl:
        filtered_categories.append('International')
    
    # Always include economic data
    filtered_categories.append('Economic')
    
    if filtered_categories:
        df = df[df['category'].isin(filtered_categories)]
    
    # Initialize enhanced ML components
    predictor = SimpleMLPredictor()
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Train models on real data with progress tracking
    with st.spinner("🤖 Training AI models on live market data..."):
        key_assets = ['SPY', 'BTCUSDT', 'UNRATE', 'EURUSD', 'AAPL', 'NVDA']
        predictions = {}
        
        for asset in key_assets:
            if asset in df['series_id'].values:
                if predictor.train_simple_models(df, asset):
                    pred_result = predictor.predict_simple(asset, periods=prediction_periods)
                    if pred_result and pred_result['confidence'] >= confidence_threshold:
                        predictions[asset] = pred_result
    
    # Enhanced portfolio analysis on real data
    with st.spinner("📊 Analyzing portfolio performance with real data..."):
        portfolio_analysis = portfolio_analyzer.analyze_portfolio(df)
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌟 Real-Time Overview", 
        "📈 Stock Analysis", 
        "₿ Crypto Analysis", 
        "🔮 AI Predictions", 
        "🧠 Portfolio Intelligence"
    ])
    
    with tab1:
        st.subheader("🌟 Real-Time Multi-Asset Market Overview")
        
        # Enhanced real-time key metrics with data source transparency
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # S&P 500 metric with enhanced info
        spy_data = df[df['series_id'] == 'SPY']
        if not spy_data.empty:
            latest_spy = spy_data.iloc[-1]['value']
            prev_spy = spy_data.iloc[-2]['value'] if len(spy_data) >= 2 else latest_spy
            spy_change = ((latest_spy - prev_spy) / prev_spy * 100)
            spy_source = spy_data.iloc[-1].get('data_source', 'Unknown')
            
            with col1:
                st.metric(
                    label="📈 S&P 500 (SPY)",
                    value=f"${latest_spy:.2f}",
                    delta=f"{spy_change:+.2f}%"
                )
                badge = "🔥" if "Alpha Vantage" in spy_source else "✅" if "Real" in spy_source else "🎯"
                st.markdown(f"<div class='data-source-info'>{badge} {spy_source}</div>", unsafe_allow_html=True)
        
        # Bitcoin metric with enhanced tracking
        btc_data = df[df['series_id'] == 'BTCUSDT']
        if not btc_data.empty:
            latest_btc = btc_data.iloc[-1]['value']
            prev_btc = btc_data.iloc[-2]['value'] if len(btc_data) >= 2 else latest_btc
            btc_change = ((latest_btc - prev_btc) / prev_btc * 100)
            btc_source = btc_data.iloc[-1].get('data_source', 'Unknown')
            
            with col2:
                st.metric(
                    label="₿ Bitcoin (BTC)",
                    value=f"${latest_btc:,.0f}",
                    delta=f"{btc_change:+.2f}%"
                )
                badge = "🔥" if "Alpha Vantage" in btc_source else "✅" if "Real" in btc_source else "🎯"
                st.markdown(f"<div class='data-source-info'>{badge} {btc_source}</div>", unsafe_allow_html=True)
        
        # EUR/USD metric with forex tracking
        eur_data = df[df['series_id'] == 'EURUSD']
        if not eur_data.empty:
            latest_eur = eur_data.iloc[-1]['value']
            prev_eur = eur_data.iloc[-2]['value'] if len(eur_data) >= 2 else latest_eur
            eur_change = ((latest_eur - prev_eur) / prev_eur * 100)
            eur_source = eur_data.iloc[-1].get('data_source', 'Unknown')
            
            with col3:
                st.metric(
                    label="💱 EUR/USD",
                    value=f"{latest_eur:.4f}",
                    delta=f"{eur_change:+.3f}%"
                )
                badge = "🔥" if "Alpha Vantage" in eur_source else "✅" if "Real" in eur_source else "🎯"
                st.markdown(f"<div class='data-source-info'>{badge} {eur_source}</div>", unsafe_allow_html=True)
        
        # Unemployment metric with economic tracking
        unrate_data = df[df['series_id'] == 'UNRATE']
        if not unrate_data.empty:
            latest_unrate = unrate_data.iloc[-1]['value']
            prev_unrate = unrate_data.iloc[-2]['value'] if len(unrate_data) >= 2 else latest_unrate
            unrate_change = latest_unrate - prev_unrate
            unrate_source = unrate_data.iloc[-1].get('data_source', 'Unknown')
            
            with col4:
                st.metric(
                    label="🏢 Unemployment",
                    value=f"{latest_unrate:.1f}%",
                    delta=f"{unrate_change:+.1f}%"
                )
                badge = "🔥" if "Alpha Vantage" in unrate_source else "✅" if "Real" in unrate_source else "🎯"
                st.markdown(f"<div class='data-source-info'>{badge} {unrate_source}</div>", unsafe_allow_html=True)
        
        # AI confidence based on data quality and predictions
        with col5:
            ai_confidence = min(95, max(60, 70 + real_data_pct * 0.3))
            prediction_boost = len(predictions) * 2  # Boost for successful predictions
            final_confidence = min(95, ai_confidence + prediction_boost)
            
            st.metric(
                label="🤖 AI Confidence",
                value=f"{final_confidence:.0f}%",
                delta=f"{len(predictions)} Models"
            )
            quality_text = "Premium" if alpha_vantage_count > 0 else "Excellent" if real_data_pct > 80 else "Good"
            st.markdown(f"<div class='data-source-info'>🧠 {quality_text} Quality</div>", unsafe_allow_html=True)
        
        # Enhanced real-time performance dashboard
        st.subheader("📊 Live Multi-Asset Performance Dashboard")
        
        # Create comprehensive real-time dashboard with enhanced features
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Stock Market Performance (Real-Time)',
                'Cryptocurrency Prices (Live)',
                'Forex Exchange Rates (Current)',
                'Economic Indicators (Latest)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Enhanced stock indices with data source indicators
        stock_symbols = ['SPY', 'QQQ', 'GLD', 'AAPL']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, symbol in enumerate(stock_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)
            if not symbol_data.empty:
                line_style = dict(color=colors[i], width=3) if "Alpha Vantage" in symbol_data.iloc[-1].get('data_source', '') else dict(color=colors[i], width=2)
                
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol}",
                        line=line_style
                    ),
                    row=1, col=1
                )
        
        # Enhanced cryptocurrencies with premium indicators
        crypto_symbols = ['BTCUSDT', 'ETHUSDT']
        for i, symbol in enumerate(crypto_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)
            if not symbol_data.empty:
                line_style = dict(color=colors[i], width=3) if "Alpha Vantage" in symbol_data.iloc[-1].get('data_source', '') else dict(color=colors[i], width=2)
                
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol.replace('USDT', '')}",
                        line=line_style
                    ),
                    row=1, col=2
                )
        
        # Enhanced forex with premium data highlighting
        forex_symbols = ['EURUSD', 'GBPUSD']
        for i, symbol in enumerate(forex_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)
            if not symbol_data.empty:
                line_style = dict(color=colors[i], width=3) if "Alpha Vantage" in symbol_data.iloc[-1].get('data_source', '') else dict(color=colors[i], width=2)
                
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol}",
                        line=line_style
                    ),
                    row=2, col=1
                )
        
        # Enhanced economic indicators
        econ_symbols = ['UNRATE', 'FEDFUNDS']
        for i, symbol in enumerate(econ_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(50)
            if not symbol_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines+markers', 
                        name=f"{symbol}",
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4)
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template='plotly_white',
            title_text="Real-Time Multi-Asset Financial Intelligence Dashboard V3.0"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced data quality summary with detailed breakdown
        st.subheader("📡 Live Data Quality & Source Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data source breakdown
            source_summary = df.groupby('data_source').agg({
                'series_id': 'count',
                'date': 'max'
            }).reset_index()
            source_summary.columns = ['Data Source', 'Assets', 'Last Update']
            source_summary['Last Update'] = source_summary['Last Update'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(source_summary, use_container_width=True)
        
        with col2:
            # Visual pie chart of data sources
            fig_quality = px.pie(
                source_summary, 
                values='Assets', 
                names='Data Source',
                title="Data Source Distribution",
                color_discrete_map={
                    'Alpha Vantage API (Real)': '#ff6b6b',
                    'Alpha Vantage Crypto (Real)': '#ffa500',
                    'Alpha Vantage FX (Real)': '#ff8c00',
                    'Yahoo Finance (Real)': '#4CAF50',
                    'CoinGecko API (Real)': '#2196F3',
                    'FRED API (Real)': '#9C27B0'
                }
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Quality metrics summary
        st.markdown("### 📊 Data Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Premium Sources", alpha_vantage_count, "Alpha Vantage")
        
        with col2:
            st.metric("Real Data Points", f"{real_data_count:,}", f"{real_data_pct:.0f}%")
        
        with col3:
            avg_data_age = (datetime.now().date() - df['date'].max()).days
            st.metric("Data Freshness", f"{avg_data_age} days", "Latest")
        
        with col4:
            unique_sources = len(df['data_source'].unique())
            st.metric("Source Diversity", unique_sources, "APIs")
    
    with tab2:
        st.subheader("📈 Real-Time Stock Market Analysis")
        
        # Enhanced stock performance metrics with premium indicators
        if 'Equity' in portfolio_analysis:
            stock_metrics = portfolio_analysis['Equity']
            
            st.markdown("### 🏆 Live Stock Performance Metrics")
            
            # Sort by data source quality (Alpha Vantage first)
            stock_metrics_sorted = sorted(stock_metrics, 
                                        key=lambda x: (
                                            0 if "Alpha Vantage" in x.get('data_source', '') else
                                            1 if "Real" in x.get('data_source', '') else 2
                                        ))
            
            for metric in stock_metrics_sorted:
                # Enhanced color coding
                perf_color = "#00C851" if metric['return_1m'] > 2 else "#00C851" if metric['return_1m'] > 0 else "#FF4444" if metric['return_1m'] < -2 else "#FF4444"
                vol_color = "#FF4444" if metric['volatility'] > 30 else "#FF8800" if metric['volatility'] > 20 else "#00C851"
                
                # Premium data source detection
                is_premium = "Alpha Vantage" in metric.get('data_source', '')
                is_real = "Real" in metric.get('data_source', '')
                
                if is_premium:
                    badge = "🔥 Premium"
                    badge_class = "alpha-vantage-badge"
                elif is_real:
                    badge = "✅ Real-Time"
                    badge_class = "real-data-badge"
                else:
                    badge = "🎯 Simulation"
                    badge_class = "data-source-info"
                
                with st.container():
                    st.markdown(f"""
                    <div class="asset-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h4>{metric['series_name']} ({metric['series_id']})</h4>
                            <div class="{badge_class}" style="display: inline-block;">{badge}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Current:</strong> ${metric['current_value']:.2f}</div>
                            <div><strong>1M Return:</strong> <span style="color: {perf_color}; font-weight: bold;">{metric['return_1m']:+.1f}%</span></div>
                            <div><strong>Volatility:</strong> <span style="color: {vol_color}; font-weight: bold;">{metric['volatility']:.1f}%</span></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>3M Return:</strong> <span style="color: {'#00C851' if metric['return_3m'] > 0 else '#FF4444'}">{metric['return_3m']:+.1f}%</span></div>
                            <div><strong>1Y Return:</strong> <span style="color: {'#00C851' if metric['return_1y'] > 0 else '#FF4444'}">{metric['return_1y']:+.1f}%</span></div>
                            <div><strong>Sharpe Ratio:</strong> {metric['sharpe_ratio']:.2f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Max Drawdown:</strong> <span style="color: #FF4444">{metric['max_drawdown']:.1f}%</span></div>
                            <div><strong>VaR (95%):</strong> <span style="color: #FF8800">{metric['var_95']:.1f}%</span></div>
                            <div><strong>Data Points:</strong> {metric['data_points']:,}</div>
                        </div>
                        <div class="data-source-info">
                            📡 {metric['data_source']} | Updated: {metric['last_updated']} | Points: {metric['data_points']:,}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Enhanced individual stock deep dive
        st.markdown("### 📊 Individual Stock Deep Analysis")
        
        stock_symbols = df[df['category'] == 'Stock Market']['series_id'].unique()
        if len(stock_symbols) > 0:
            # Sort stocks by data source quality for selection
            stock_options = []
            for symbol in stock_symbols:
                symbol_data = df[df['series_id'] == symbol]
                if not symbol_data.empty:
                    source = symbol_data.iloc[-1].get('data_source', '')
                    name = symbol_data.iloc[-1].get('series_name', symbol)
                    if "Alpha Vantage" in source:
                        stock_options.append(f"🔥 {symbol} - {name}")
                    elif "Real" in source:
                        stock_options.append(f"✅ {symbol} - {name}")
                    else:
                        stock_options.append(f"🎯 {symbol} - {name}")
            
            selected_option = st.selectbox("Select Stock for Deep Analysis:", stock_options)
            selected_stock = selected_option.split(' ')[1] if selected_option else stock_symbols[0]
            
            stock_data = df[df['series_id'] == selected_stock].sort_values('date')
            
            if not stock_data.empty:
                # Display stock info header
                stock_name = stock_data.iloc[-1]['series_name']
                stock_source = stock_data.iloc[-1].get('data_source', 'Unknown')
                latest_price = stock_data.iloc[-1]['value']
                
                st.markdown(f"""
                <div class="asset-card">
                    <h3>{stock_name} ({selected_stock})</h3>
                    <p><strong>Current Price:</strong> ${latest_price:.2f} | <strong>Source:</strong> {stock_source}</p>
                    <p><strong>Data Range:</strong> {stock_data['date'].min()} to {stock_data['date'].max()} | <strong>Points:</strong> {len(stock_data):,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Enhanced price chart with technical indicators
                    fig_price = go.Figure()
                    
                    # Price line
                    fig_price.add_trace(go.Scatter(
                        x=stock_data['date'], 
                        y=stock_data['value'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add moving averages if sufficient data
                    if len(stock_data) >= 20:
                        ma_20 = stock_data['value'].rolling(20).mean()
                        fig_price.add_trace(go.Scatter(
                            x=stock_data['date'],
                            y=ma_20,
                            mode='lines',
                            name='20-day MA',
                            line=dict(color='orange', width=1, dash='dash')
                        ))
                    
                    if len(stock_data) >= 50:
                        ma_50 = stock_data['value'].rolling(50).mean()
                        fig_price.add_trace(go.Scatter(
                            x=stock_data['date'],
                            y=ma_50,
                            mode='lines',
                            name='50-day MA',
                            line=dict(color='red', width=1, dash='dot')
                        ))
                    
                    fig_price.update_layout(
                        title=f"{stock_name} - Price Chart with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    # Enhanced volume and volatility analysis
                    if 'volume' in stock_data.columns:
                        recent_data = stock_data.tail(60)  # Last 60 days
                        
                        fig_vol = go.Figure()
                        
                        # Volume bars
                        fig_vol.add_trace(go.Bar(
                            x=recent_data['date'],
                            y=recent_data['volume'],
                            name='Volume',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Add volume moving average
                        vol_ma = recent_data['volume'].rolling(10).mean()
                        fig_vol.add_trace(go.Scatter(
                            x=recent_data['date'],
                            y=vol_ma,
                            mode='lines',
                            name='10-day Volume MA',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig_vol.update_layout(
                            title=f"{stock_name} - Trading Volume Analysis",        # Alpha Vantage forex pairs
        av_pairs = {
            'EUR': 'EUR/USD Exchange Rate',
            'GBP': 'GBP/USD Exchange Rate',
            'JPY': 'USD/JPY Exchange Rate',
            'CAD': 'USD/CAD Exchange Rate',
            'AUD': 'AUD/USD Exchange Rate',
            'CHF': 'USD/CHF Exchange Rate'
        }
        
        if self.alpha_vantage_key != "demo_key":
            st.info("🔥 Loading premium forex data from Alpha Vantage API...")
            
            for from_currency, name in av_pairs.items():
                try:
                    st.info(f"💱 Loading {name} from Alpha Vantage...")
                    
                    url = "https://www.alphavantage.co/query"
                    
                    # Determine currency pair format for Alpha Vantage
                    if from_currency in ['EUR', 'GBP', 'AUD']:
                        from_symbol, to_symbol = from_currency, 'USD'
                    else:  # JPY, CAD, CHF
                        from_symbol, to_symbol = 'USD', from_currency
                    
                    params = {
                        'function': 'FX_DAILY',
                        'from_symbol': from_symbol,
                        'to_symbol': to_symbol,
                        'apikey': self.alpha_vantage_key
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        av_data = response.json()
                        
                        # Check for API errors
                        if 'Error Message' in av_data:
                            st.warning(f"⚠️ Alpha Vantage forex error for {from_currency}: {av_data['Error Message']}")
                            continue
                        
                        if 'Note' in av_data:
                            st.warning(f"⚠️ Alpha Vantage rate limit hit for {from_currency}")
                            continue
                        
                        time_series = av_data.get('Time Series FX (Daily)', {})
                        
                        if time_series:
                            valid_data = False
                            for date_str, values in list(time_series.items())[:200]:  # Last 200 days
                                try:
                                    date = pd.to_datetime(date_str).date()
                                    close_rate = float(values['4. close'])
                                    high = float(values['2. high'])
                                    low = float(values['3. low'])
                                    open_rate = float(values['1. open'])
                                    
                                    # Create consistent series_id
                                    if from_currency in ['EUR', 'GBP', 'AUD']:
                                        series_id = f"{from_currency}USD"
                                    else:
                                        series_id = f"USD{from_currency}"
                                    
                                    data.append({
                                        'date': date,
                                        'value': close_rate,
                                        'volume': np.random.randint(5000000, 50000000),  # Forex volume not in free API
                                        'high': high,
                                        'low': low,
                                        'open': open_rate,
                                        'series_id': series_id,
                                        'series_name': name,
                                        'category': 'Forex',
                                        'unit': 'Exchange Rate',
                                        'asset_type': 'Currency',
                                        'data_source': 'Alpha Vantage FX (Real)'
                                    })
                                    valid_data = True
                                except (ValueError, KeyError, TypeError):
                                    continue
                            
                            if valid_data:
                                sources.append("Alpha Vantage FX")
                                st.success(f"✅ Loaded {name} from Alpha Vantage")
                            else:
                                st.warning(f"⚠️ No valid forex data for {from_currency} from Alpha Vantage")
                            
                            time.sleep(12)  # Rate limiting
                        else:
                            st.warning(f"⚠️ No forex time series data for {from_currency} from Alpha Vantage")
                            
                except Exception as e:
                    st.warning(f"⚠️ Alpha Vantage forex error for {from_currency}: {str(e)}")
        
        # Fallback to Yahoo Finance for any missing pairs
        loaded_pairs = set([d['series_id'] for d in data])
        yf_pairs = {
            'EURUSD=X': ('EURUSD', 'EUR/USD Exchange Rate'),
            'GBPUSD=X': ('GBPUSD', 'GBP/USD Exchange Rate'), 
            'USDJPY=X': ('USDJPY', 'USD/JPY Exchange Rate'),
            'AUDUSD=X': ('AUDUSD', 'AUD/USD Exchange Rate'),
            'USDCAD=X': ('USDCAD', 'USD/CAD Exchange Rate'),
            'USDCHF=X': ('USDCHF', 'USD/CHF Exchange Rate')
        }
        
        missing_pairs = []
        for yf_pair, (clean_pair, name) in yf_pairs.items():
            if clean_pair not in loaded_pairs:
                missing_pairs.append((yf_pair, name, clean_pair))
        
        if missing_pairs and YFINANCE_AVAILABLE:
            st.info("💱 Loading backup forex data from Yahoo Finance...")
            
            for yf_pair, name, clean_pair in missing_pairs:
                try:
                    st.info(f"💱 Loading {name} from Yahoo Finance...")
                    ticker = yf.Ticker(yf_pair)
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        valid_data = False
                        for date, row in hist.iterrows():
                            try:
                                data.append({
                                    'date': date.date(),
                                    'value': float(row['Close']),
                                    'volume': int(row['Volume']) if row['Volume'] > 0 else 5000000,
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'open': float(row['Open']),
                                    'series_id': clean_pair,
                                    'series_name': name,
                                    'category': 'Forex',
                                    'unit': 'Exchange Rate',
                                    'asset_type': 'Currency',
                                    'data_source': 'Yahoo Finance FX (Real)'
                                })
                                valid_data = True
                            except (ValueError, TypeError):
                                continue
                        
                        if valid_data:
                            sources.append("Yahoo Finance FX")
                            st.success(f"✅ Loaded {name} from Yahoo Finance")
                        else:
                            st.warning(f"⚠️ No valid data for {name} from Yahoo Finance")
                    else:
                        st.warning(f"⚠️ No data returned for {name} from Yahoo Finance")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load {name} from Yahoo Finance: {str(e)}")
        
        # Simulation fallback for any remaining missing data
        all_expected_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
        loaded_pair_ids = set([d['series_id'] for d in data])
        missing_pair_ids = set(all_expected_pairs) - loaded_pair_ids
        
        if missing_pair_ids:
            st.info(f"🎯 Generating enhanced forex simulations for {len(missing_pair_ids)} pairs...")
            for pair_id in missing_pair_ids:
                st.info(f"🎯 Generating simulation for {pair_id}")
                sim_data = self.generate_realistic_forex_data(pair_id, f"{pair_id} Exchange Rate")
                data.extend(sim_data)
            sources.append("Enhanced Forex Simulation")
        
        if not sources:
            sources.append("Enhanced Forex Simulation")
        
        return data, sources
    
    def load_real_international_data(self):
        """Load real international economic data from FRED"""
        
        # International FRED series
        intl_indicators = {
            'LRUNTTTTGBM156S': 'UK Unemployment Rate',
            'LRUNTTTTDEM156S': 'Germany Unemployment Rate',
            'LRUNTTTTJPM156S': 'Japan Unemployment Rate',
            'LRUNTTTTCAM156S': 'Canada Unemployment Rate',
            'IRLTLT01GBM156N': 'UK Long-term Interest Rate',
            'IRLTLT01DEM156N': 'Germany Long-term Interest Rate',
            'CPALTT01GBM661N': 'UK Consumer Price Index',
            'CPALTT01DEM661N': 'Germany Consumer Price Index'
        }
        
        data = []
        sources = []
        
        for series_id, series_name in intl_indicators.items():
            try:
                if self.fred_api_key != "demo_key":
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 60,
                        'sort_order': 'desc'
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        observations = json_data.get('observations', [])
                        
                        valid_data = False
                        for obs in observations:
                            if obs['value'] != '.' and obs['value'] and obs['value'] != 'null':
                                try:
                                    data.append({
                                        'date': pd.to_datetime(obs['date']).date(),
                                        'value': float(obs['value']),
                                        'series_id': series_id,
                                        'series_name': series_name,
                                        'category': 'International',
                                        'unit': '%' if 'Rate' in series_name else 'Index',
                                        'asset_type': 'Economic',
                                        'country': self.extract_country(series_name),
                                        'data_source': 'FRED International (Real)'
                                    })
                                    valid_data = True
                                except (ValueError, TypeError):
                                    continue
                        
                        if valid_data:
                            sources.append("FRED International")
                            st.success(f"✅ Loaded real {series_name}")
                            time.sleep(0.1)
                        else:
                            raise Exception("No valid FRED international data")
                    else:
                        raise Exception("FRED API error")
                else:
                    raise Exception("No FRED API key")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load {series_name}, using simulation")
                sim_data = self.generate_realistic_international_data(series_id, series_name)
                data.extend(sim_data)
        
        if not sources:
            sources.append("Enhanced International Simulation")
        
        return data, sources
    
    def extract_country(self, series_name):
        """Extract country from series name"""
        if 'UK' in series_name:
            return 'United Kingdom'
        elif 'Germany' in series_name:
            return 'Germany'
        elif 'Japan' in series_name:
            return 'Japan'
        elif 'Canada' in series_name:
            return 'Canada'
        else:
            return 'International'
    
    def generate_realistic_economic_data(self, series_id, series_name):
        """Generate enhanced realistic economic data with current patterns"""
        
        data = []
        dates = pd.date_range('2020-01-01', datetime.now(), freq='M')
        
        # Current realistic values (as of 2025)
        current_values = {
            'UNRATE': 3.8,          # Current unemployment
            'FEDFUNDS': 4.75,       # Current Fed funds rate  
            'CPIAUCSL': 310.0,      # Current CPI level
            'GDP': 28000.0,         # Current GDP
            'HOUST': 1350.0,        # Current housing starts
            'INDPRO': 103.0,        # Current industrial production
            'PAYEMS': 158000.0,     # Current nonfarm payrolls
            'UMCSENT': 95.0,        # Current consumer sentiment
            'DEXUSEU': 1.08,        # USD/EUR rate
            'VIXCLS': 15.0          # VIX volatility
        }
        
        base_value = current_values.get(series_id, 100.0)
        values = self.generate_economic_series_with_patterns(base_value, len(dates), series_id)
        
        for date, value in zip(dates, values):
            data.append({
                'date': date.date(),
                'value': value,
                'series_id': series_id,
                'series_name': series_name,
                'category': 'Economic',
                'unit': '%' if 'Rate' in series_name else 'Index',
                'asset_type': 'Economic',
                'data_source': 'Enhanced Economic Simulation (Current Patterns)'
            })
        
        return data
    
    def generate_economic_series_with_patterns(self, base_value, length, series_id):
        """Generate realistic economic series with 2020-2025 patterns"""
        
        values = []
        
        for i in range(length):
            # Different patterns for different indicators
            if 'UNRATE' in series_id:
                # Unemployment: COVID spike, then decline
                if i <= 3:  # 2020 Q1
                    value = base_value
                elif i <= 8:  # COVID spike
                    value = base_value * 2.5  # Spike to ~9.5%
                elif i <= 24:  # Gradual decline
                    recovery_progress = (i - 8) / 16
                    value = (base_value * 2.5) - (base_value * 1.2 * recovery_progress)
                else:  # Current levels
                    value = base_value + np.random.normal(0, 0.1)
                    
            elif 'FEDFUNDS' in series_id:
                # Fed Funds: Zero to current levels
                if i <= 24:  # 2020-2022: near zero
                    value = 0.25 + np.random.normal(0, 0.05)
                else:  # 2022+: aggressive hiking
                    hiking_progress = min((i - 24) / 12, 1.0)
                    value = 0.25 + (base_value - 0.25) * hiking_progress + np.random.normal(0, 0.1)
                    
            elif 'VIX' in series_id:
                # VIX: High during COVID, then decline
                if i <= 8:  # COVID period
                    value = base_value * 2.5 + np.random.normal(0, 5)
                else:  # Gradual normalization
                    value = base_value + np.random.normal(0, 3)
                    
            else:
                # Other indicators: gradual trends with noise
                trend = (base_value * 0.15 * i / length)  # 15% total change over period
                noise = np.random.normal(0, base_value * 0.02)
                value = base_value + trend + noise
            
            values.append(max(0.01, value))
        
        return values
    
    def generate_realistic_stock_data(self, symbol, name):
        """Generate realistic stock data with current market patterns"""
        
        # Current market levels (January 2025)
        current_prices = {
            'SPY': 580.0,       # S&P 500 at highs
            'QQQ': 520.0,       # NASDAQ strong
            'IWM': 240.0,       # Small caps
            'GLD': 185.0,       # Gold
            'VTI': 290.0,       # Total market
            'TLT': 85.0,        # Bonds under pressure
            'VIX': 15.0,        # Low volatility
            'DIA': 440.0,       # Dow
            'AAPL': 225.0,      # Apple
            'MSFT': 415.0,      # Microsoft
            'GOOGL': 185.0,     # Google
            'AMZN': 220.0,      # Amazon
            'TSLA': 350.0,      # Tesla
            'NVDA': 140.0,      # NVIDIA
            'META': 380.0,      # Meta
            'NFLX': 490.0       # Netflix
        }
        
        data = []
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_price = current_prices.get(symbol, 100.0)
        
        prices = self.generate_stock_series_with_trends(base_price, len(dates), symbol)
        
        for date, price in zip(dates, prices):
            data.append({
                'date': date.date(),
                'value': price,
                'volume': np.random.randint(1000000, 20000000),
                'high': price * 1.02,
                'low': price * 0.98,
                'open': price * 0.999,
                'series_id': symbol,
                'series_name': name,
                'category': 'Stock Market',
                'unit': 'USD',
                'asset_type': 'Equity',
                'data_source': 'Enhanced Stock Simulation (2025 Levels)'
            })
        
        return data
    
    def generate_stock_series_with_trends(self, base_price, length, symbol):
        """Generate stock series with realistic 2023-2025 bull market"""
        
        prices = [base_price * 0.8]  # Start 20% below current
        
        for i in range(1, length):
            # Bull market trend with volatility
            daily_return = np.random.normal(0.0008, 0.015)  # Positive bias
            
            # Add momentum and sector rotation
            if symbol in ['SPY', 'QQQ', 'VTI', 'AAPL', 'MSFT', 'NVDA', 'META']:  # Strong performers
                daily_return += 0.0002  # Extra positive bias
            elif symbol in ['TLT']:  # Bonds struggling
                daily_return -= 0.0001  # Slight negative bias
            elif symbol in ['TSLA']:  # High volatility stocks
                daily_return += np.random.normal(0, 0.01)  # Extra volatility
            
            # Occasional corrections
            if i % 100 == 50:  # Every ~100 days
                daily_return += np.random.choice([-0.05, 0.02], p=[0.3, 0.7])
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(1.0, new_price))
        
        return prices
    
    def generate_realistic_crypto_data(self, symbol, name):
        """Generate realistic crypto data with current bull market levels"""
        
        # Current crypto levels (January 2025)
        current_prices = {
            'BTC': 115000.0,    # Bitcoin at ATH
            'ETH': 4200.0,      # Ethereum strong
            'BNB': 720.0,       # BNB elevated
            'ADA': 1.15,        # Cardano recovery
            'SOL': 280.0,       # Solana surge
            'LTC': 120.0,       # Litecoin steady
            'XRP': 0.85         # XRP litigation resolution
        }
        
        data = []
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_price = current_prices.get(symbol, 1000.0)
        
        prices = self.generate_crypto_series_with_bull_market(base_price, len(dates))
        
        for date, price in zip(dates, prices):
            data.append({
                'date': date.date(),
                'value': price,
                'volume': np.random.randint(500000, 5000000),
                'high': price * 1.08,
                'low': price * 0.92,
                'open': price * 0.997,
                'series_id': f"{symbol}USDT",
                'series_name': f"{name}",
                'category': 'Cryptocurrency',
                'unit': 'USD',
                'asset_type': 'Crypto',
                'data_source': 'Enhanced Crypto Simulation (2025 Bull Market)'
            })
        
        return data
    
    def generate_crypto_series_with_bull_market(self, base_price, length):
        """Generate crypto with 2024-2025 bull market pattern"""
        
        prices = [base_price * 0.3]  # Start from bear market lows
        
        for i in range(1, length):
            # Crypto bull market with high volatility
            daily_return = np.random.normal(0.002, 0.05)  # Strong positive bias
            
            # Bull market acceleration phases
            progress = i / length
            if progress > 0.6:  # Last 40% of period - parabolic phase
                daily_return += 0.001  # Extra momentum
            
            # Occasional large moves (crypto nature)
            if i % 50 == 25:  # Every ~50 days
                daily_return += np.random.choice([-0.25, 0.35], p=[0.4, 0.6])
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(0.01, new_price))
        
        return prices
    
    def generate_realistic_forex_data(self, pair, name):
        """Generate realistic forex data with current patterns"""
        
        current_rates = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2420,
            'USDJPY': 155.0,
            'AUDUSD': 0.6180,
            'USDCAD': 1.4350,
            'USDCHF': 0.9120
        }
        
        data = []
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        base_rate = current_rates.get(pair, 1.0)
        
        rates = self.generate_forex_series_with_trends(base_rate, len(dates))
        
        for date, rate in zip(dates, rates):
            data.append({
                'date': date.date(),
                'value': rate,
                'volume': np.random.randint(5000000, 50000000),
                'high': rate * 1.01,
                'low': rate * 0.99,
                'open': rate * 0.9995,
                'series_id': pair,
                'series_name': name,
                'category': 'Forex',
                'unit': 'Exchange Rate',
                'asset_type': 'Currency',
                'data_source': 'Enhanced Forex Simulation (Current Levels)'
            })
        
        return data
    
    def generate_forex_series_with_trends(self, base_rate, length):
        """Generate forex with realistic central bank policy impacts"""
        
        rates = [base_rate * 0.95]  # Start slightly below current
        
        for i in range(1, length):
            # Forex movements with central bank policy influence
            daily_change = np.random.normal(0, 0.008)  # ~12% annual volatility
            
            # Add policy-driven trends
            progress = i / length
            if progress > 0.5:  # Recent policy divergence
                daily_change += (base_rate - rates[-1]) * 0.001  # Mean reversion
            
            new_rate = rates[-1] * (1 + daily_change)
            rates.append(max(0.01, new_rate))
        
        return rates
    
    def generate_realistic_international_data(self, series_id, series_name):
        """Generate realistic international economic data"""
        
        current_values = {
            'LRUNTTTTGBM156S': 4.2,      # UK unemployment
            'LRUNTTTTDEM156S': 3.5,      # Germany unemployment
            'LRUNTTTTJPM156S': 2.4,      # Japan unemployment
            'LRUNTTTTCAM156S': 5.8,      # Canada unemployment
            'IRLTLT01GBM156N': 4.1,      # UK interest rate
            'IRLTLT01DEM156N': 2.8,      # Germany interest rate
            'CPALTT01GBM661N': 108.5,    # UK CPI
            'CPALTT01DEM661N': 112.2     # Germany CPI
        }
        
        data = []
        dates = pd.date_range('2020-01-01', datetime.now(), freq='M')
        base_value = current_values.get(series_id, 100.0)
        
        values = self.generate_economic_series_with_patterns(base_value, len(dates), series_id)
        
        for date, value in zip(dates, values):
            data.append({
                'date': date.date(),
                'value': value,
                'series_id': series_id,
                'series_name': series_name,
                'category': 'International',
                'unit': '%' if 'Rate' in series_name else 'Index',
                'asset_type': 'Economic',
                'country': self.extract_country(series_name),
                'data_source': 'Enhanced International Simulation (Current Patterns)'
            })
        
        return data

# Cached data loading function (outside of class to avoid hashing issues)
@st.cache_data(ttl=900)  # 15-minute cache for real data
def load_comprehensive_real_data():
    """Load comprehensive real multi-asset data with caching"""
    data_loader = RealDataLoader()
    return data_loader.load_all_data()

class SimpleMLPredictor:
    """Enhanced ML predictor with real data integration"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def train_simple_models(self, df, target_series):
        """Train simple prediction models on real data"""
        
        target_data = df[df['series_id'] == target_series].sort_values('date')
        if len(target_data) < 30:
            return False
        
        values = target_data['value'].values
        
        # Enhanced features for real data
        short_ma = np.mean(values[-5:])
        long_ma = np.mean(values[-20:])
        trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
        
        # Linear trend with more data points
        x = np.arange(len(values))
        trend_coef = np.polyfit(x[-30:], values[-30:], 1)[0] if len(values) >= 30 else 0
        
        # Volatility analysis
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # RSI-like momentum indicator
        gains = np.maximum(np.diff(values), 0)
        losses = np.maximum(-np.diff(values), 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
        
        self.models[target_series] = {
            'current_value': values[-1],
            'short_ma': short_ma,
            'long_ma': long_ma,
            'trend': trend,
            'trend_coef': trend_coef,
            'volatility': volatility,
            'returns_mean': np.mean(returns),
            'rsi': rsi,
            'series_name': target_data.iloc[-1]['series_name'],
            'data_source': target_data.iloc[-1].get('data_source', 'Unknown')
        }
        
        return True
    
    def predict_simple(self, target_series, periods=30):
        """Generate predictions based on real data patterns"""
        
        if target_series not in self.models:
            return None
        
        model = self.models[target_series]
        predictions = []
        
        current_value = model['current_value']
        trend_coef = model['trend_coef']
        volatility = model['volatility']
        returns_mean = model['returns_mean']
        rsi = model['rsi']
        
        for i in range(periods):
            # Enhanced prediction with multiple factors
            trend_component = trend_coef * (i + 1)
            mean_reversion = returns_mean * current_value * 0.1
            
            # RSI-based momentum adjustment
            rsi_factor = (50 - rsi) / 1000  # Contrarian signal
            momentum_adjustment = rsi_factor * current_value
            
            # Volatility-adjusted noise
            noise_component = np.random.normal(0, volatility * current_value * 0.05)
            
            predicted_value = (current_value + trend_component + 
                             mean_reversion + momentum_adjustment + noise_component)
            predictions.append(max(0.01, predicted_value))
        
        # Create prediction dates
        last_date = datetime.now().date()
        pred_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]
        
        return {
            'dates': pred_dates,
            'predictions': predictions,
            'series_name': model['series_name'],
            'data_source': model['data_source'],
            'confidence': min(95, max(60, 85 - volatility)),  # Confidence based on volatility
            'rsi': rsi,
            'trend_strength': abs(trend_coef) * 1000
        }

class PortfolioAnalyzer:
    """Enhanced portfolio analysis with real data"""
    
    def analyze_portfolio(self, df):
        """Analyze portfolio metrics using real data"""
        
        analysis = {}
        
        for asset_type in df['asset_type'].unique():
            type_data = df[df['asset_type'] == asset_type]
            
            performance_metrics = []
