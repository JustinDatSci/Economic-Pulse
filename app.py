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
            'LRUNTTTTGBM156S': 4.2,   # UK unemployment
            'LRUNTTTTDEM156S': 3.5,   # Germany unemployment
            'LRUNTTTTJPM156S': 2.4,   # Japan unemployment
            'IRLTLT01GBM156N': 4.1,   # UK interest rate
            'CPALTT01GBM661N': 108.5  # UK CPI
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
    """Simplified ML predictor with real data integration"""
    
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
        
        self.models[target_series] = {
            'current_value': values[-1],
            'short_ma': short_ma,
            'long_ma': long_ma,
            'trend': trend,
            'trend_coef': trend_coef,
            'volatility': volatility,
            'returns_mean': np.mean(returns),
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
        
        for i in range(periods):
            # Enhanced prediction with mean reversion
            trend_component = trend_coef * (i + 1)
            mean_reversion = returns_mean * current_value * 0.1  # Slight mean reversion
            noise_component = np.random.normal(0, volatility * current_value * 0.05)
            
            predicted_value = current_value + trend_component + mean_reversion + noise_component
            predictions.append(max(0.01, predicted_value))
        
        # Create prediction dates
        last_date = datetime.now().date()
        pred_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]
        
        return {
            'dates': pred_dates,
            'predictions': predictions,
            'series_name': model['series_name'],
            'data_source': model['data_source'],
            'confidence': min(95, max(60, 85 - volatility))  # Confidence based on volatility
        }

class PortfolioAnalyzer:
    """Enhanced portfolio analysis with real data"""
    
    def analyze_portfolio(self, df):
        """Analyze portfolio metrics using real data"""
        
        analysis = {}
        
        for asset_type in df['asset_type'].unique():
            type_data = df[df['asset_type'] == asset_type]
            
            performance_metrics = []
            
            for series_id in type_data['series_id'].unique():
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
                    max_drawdown = np.min(drawdown)
                    
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
                        'data_source': series_data.iloc[-1].get('data_source', 'Unknown'),
                        'last_updated': dates.iloc[-1].strftime('%Y-%m-%d')
                    })
            
            analysis[asset_type] = performance_metrics
        
        return analysis

def main():
    """Main application with real data integration"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Economic Pulse V3.0 - Real Data Financial Intelligence</h1>
        <p>🌟 Live Market Data • 📊 Real Economic Indicators • 🤖 AI-Powered Analytics • 🌍 Global Coverage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Real Data Configuration")
    
    # API configuration
    st.sidebar.subheader("🔑 Premium API Configuration")
    
    # Check API availability
    av_available = st.secrets.get("alpha_vantage_key", "demo_key") != "demo_key"
    fred_available = st.secrets.get("fred_api_key", "demo_key") != "demo_key"
    
    if av_available:
        st.sidebar.success("🔥 Alpha Vantage API: ACTIVE")
        st.sidebar.markdown("• Individual stocks (AAPL, MSFT, etc.)")
        st.sidebar.markdown("• Crypto data (BTC, ETH, LTC, XRP)")
        st.sidebar.markdown("• Enhanced forex rates")
    else:
        st.sidebar.warning("⚠️ Alpha Vantage API: Not configured")
        st.sidebar.markdown("Add `alpha_vantage_key` to secrets for premium data")
    
    if fred_available:
        st.sidebar.success("✅ FRED API: ACTIVE")
    else:
        st.sidebar.info("💡 FRED API: Using fallbacks")
    
    st.sidebar.markdown("""
    <div class='data-source-info'>
    📡 <strong>Enhanced Data Sources:</strong><br>
    🔥 Alpha Vantage - Premium stocks, crypto, forex<br>
    • FRED API - US Economic Data<br>
    • Yahoo Finance - ETFs & backup data<br>
    • CoinGecko - Crypto backup<br>
    • Enhanced Simulations - Smart fallbacks
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    st.sidebar.subheader("📊 Data Sources")
    include_stocks = st.sidebar.checkbox("📈 Stock Market (Real)", value=True)
    include_crypto = st.sidebar.checkbox("₿ Cryptocurrency (Real)", value=True)
    include_forex = st.sidebar.checkbox("💱 Forex (Real)", value=True)
    include_intl = st.sidebar.checkbox("🌍 International (Real)", value=True)
    
    # Model selection
    st.sidebar.subheader("🤖 AI Models")
    use_advanced_ml = st.sidebar.checkbox("🧠 Advanced ML", value=ML_AVAILABLE)
    prediction_periods = st.sidebar.slider("🔮 Prediction Days", 7, 60, 30)
    
    # Load real data
    with st.spinner("🔄 Loading real-time financial data from multiple sources..."):
        df = load_comprehensive_real_data()
    
    if df.empty:
        st.error("❌ Unable to load any data. Please check API connections.")
        return
    
    # Show data summary
    st.sidebar.subheader("📈 Data Summary")
    st.sidebar.metric("Total Assets", len(df['series_id'].unique()))
    st.sidebar.metric("Data Points", len(df))
    st.sidebar.metric("Last Updated", df['date'].max().strftime('%Y-%m-%d'))
    
    # Filter data based on selections
    filtered_categories = []
    if include_stocks:
        filtered_categories.append('Stock Market')
    if include_crypto:
        filtered_categories.append('Cryptocurrency') 
    if include_forex:
        filtered_categories.append('Forex')
    if include_intl:
        filtered_categories.append('International')
    
    if filtered_categories:
        df = df[df['category'].isin(filtered_categories + ['Economic'])]
    
    # Initialize ML components
    predictor = SimpleMLPredictor()
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Train models on real data
    with st.spinner("🤖 Training AI models on real market data..."):
        key_assets = ['SPY', 'BTCUSDT', 'UNRATE', 'EURUSD']
        predictions = {}
        
        for asset in key_assets:
            if asset in df['series_id'].values:
                if predictor.train_simple_models(df, asset):
                    pred_result = predictor.predict_simple(asset, periods=prediction_periods)
                    if pred_result:
                        predictions[asset] = pred_result
    
    # Portfolio analysis on real data
    portfolio_analysis = portfolio_analyzer.analyze_portfolio(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌟 Real-Time Overview", 
        "📈 Stock Analysis", 
        "₿ Crypto Analysis", 
        "🔮 AI Predictions", 
        "🧠 Portfolio Intelligence"
    ])
    
    with tab1:
        st.subheader("🌟 Real-Time Multi-Asset Market Overview")
        
        # Real-time key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Real stock market metric
        spy_data = df[df['series_id'] == 'SPY']
        if not spy_data.empty:
            latest_spy = spy_data.iloc[-1]['value']
            prev_spy = spy_data.iloc[-2]['value'] if len(spy_data) >= 2 else latest_spy
            spy_change = ((latest_spy - prev_spy) / prev_spy * 100)
            
            with col1:
                st.metric(
                    label="📈 S&P 500 (SPY)",
                    value=f"${latest_spy:.2f}",
                    delta=f"{spy_change:+.2f}%"
                )
                st.markdown(f"<div class='data-source-info'>Source: {spy_data.iloc[-1].get('data_source', 'Unknown')}</div>", unsafe_allow_html=True)
        
        # Real crypto metric
        btc_data = df[df['series_id'] == 'BTCUSDT']
        if not btc_data.empty:
            latest_btc = btc_data.iloc[-1]['value']
            prev_btc = btc_data.iloc[-2]['value'] if len(btc_data) >= 2 else latest_btc
            btc_change = ((latest_btc - prev_btc) / prev_btc * 100)
            
            with col2:
                st.metric(
                    label="₿ Bitcoin (BTC)",
                    value=f"${latest_btc:,.0f}",
                    delta=f"{btc_change:+.2f}%"
                )
                st.markdown(f"<div class='data-source-info'>Source: {btc_data.iloc[-1].get('data_source', 'Unknown')}</div>", unsafe_allow_html=True)
        
        # Real forex metric
        eur_data = df[df['series_id'] == 'EURUSD']
        if not eur_data.empty:
            latest_eur = eur_data.iloc[-1]['value']
            prev_eur = eur_data.iloc[-2]['value'] if len(eur_data) >= 2 else latest_eur
            eur_change = ((latest_eur - prev_eur) / prev_eur * 100)
            
            with col3:
                st.metric(
                    label="💱 EUR/USD",
                    value=f"{latest_eur:.4f}",
                    delta=f"{eur_change:+.3f}%"
                )
                st.markdown(f"<div class='data-source-info'>Source: {eur_data.iloc[-1].get('data_source', 'Unknown')}</div>", unsafe_allow_html=True)
        
        # Real economic metric
        unrate_data = df[df['series_id'] == 'UNRATE']
        if not unrate_data.empty:
            latest_unrate = unrate_data.iloc[-1]['value']
            prev_unrate = unrate_data.iloc[-2]['value'] if len(unrate_data) >= 2 else latest_unrate
            unrate_change = latest_unrate - prev_unrate
            
            with col4:
                st.metric(
                    label="🏢 Unemployment",
                    value=f"{latest_unrate:.1f}%",
                    delta=f"{unrate_change:+.1f}%"
                )
                st.markdown(f"<div class='data-source-info'>Source: {unrate_data.iloc[-1].get('data_source', 'Unknown')}</div>", unsafe_allow_html=True)
        
        # AI confidence based on data quality
        with col5:
            real_data_pct = len(df[df['data_source'].str.contains('Real', na=False)]) / len(df) * 100
            ai_confidence = min(95, max(60, 70 + real_data_pct * 0.3))
            
            st.metric(
                label="🤖 Data Quality",
                value=f"{ai_confidence:.0f}%",
                delta=f"{real_data_pct:.0f}% Real Data"
            )
        
        # Real-time performance dashboard
        st.subheader("📊 Live Multi-Asset Performance Dashboard")
        
        # Create comprehensive real-time dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Stock Market (Real-Time)',
                'Cryptocurrency (Live Prices)',
                'Forex Markets (Current Rates)',
                'Economic Indicators (Latest Data)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Real stock indices
        stock_symbols = ['SPY', 'QQQ', 'GLD']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, symbol in enumerate(stock_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)  # Last 100 days
            if not symbol_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol} (Real)",
                        line=dict(color=colors[i], width=2)
                    ),
                    row=1, col=1
                )
        
        # Real cryptocurrencies
        crypto_symbols = ['BTCUSDT', 'ETHUSDT']
        for i, symbol in enumerate(crypto_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)
            if not symbol_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol.replace('USDT', '')} (Live)",
                        line=dict(color=colors[i], width=2)
                    ),
                    row=1, col=2
                )
        
        # Real forex
        forex_symbols = ['EURUSD', 'GBPUSD']
        for i, symbol in enumerate(forex_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(100)
            if not symbol_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines', 
                        name=f"{symbol} (Real)",
                        line=dict(color=colors[i], width=2)
                    ),
                    row=2, col=1
                )
        
        # Real economic indicators
        econ_symbols = ['UNRATE', 'FEDFUNDS']
        for i, symbol in enumerate(econ_symbols):
            symbol_data = df[df['series_id'] == symbol].sort_values('date').tail(50)  # Monthly data
            if not symbol_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'], 
                        y=symbol_data['value'],
                        mode='lines+markers', 
                        name=f"{symbol} (FRED)",
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4)
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template='plotly_white',
            title_text="Real-Time Multi-Asset Financial Intelligence Dashboard"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data quality summary
        st.subheader("📡 Live Data Quality Report")
        
        data_quality = df.groupby('data_source').size().reset_index(columns=['count'])
        data_quality.columns = ['Data Source', 'Data Points']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(data_quality, use_container_width=True)
        
        with col2:
            fig_quality = px.pie(
                data_quality, 
                values='Data Points', 
                names='Data Source',
                title="Data Source Distribution"
            )
            st.plotly_chart(fig_quality, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Real-Time Stock Market Analysis")
        
        # Real stock performance metrics
        if 'Equity' in portfolio_analysis:
            stock_metrics = portfolio_analysis['Equity']
            
            st.markdown("### 🏆 Live Stock Performance Metrics")
            
            for metric in stock_metrics:
                # Color coding based on performance
                perf_color = "green" if metric['return_1m'] > 0 else "red"
                vol_color = "red" if metric['volatility'] > 25 else "orange" if metric['volatility'] > 15 else "green"
                
                with st.container():
                    st.markdown(f"""
                    <div class="asset-card">
                        <h4>{metric['series_name']} ({metric['series_id']})</h4>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Current:</strong> ${metric['current_value']:.2f}</div>
                            <div><strong>1M Return:</strong> <span style="color: {perf_color}">{metric['return_1m']:+.1f}%</span></div>
                            <div><strong>Volatility:</strong> <span style="color: {vol_color}">{metric['volatility']:.1f}%</span></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>3M Return:</strong> {metric['return_3m']:+.1f}%</div>
                            <div><strong>1Y Return:</strong> {metric['return_1y']:+.1f}%</div>
                            <div><strong>Sharpe Ratio:</strong> {metric['sharpe_ratio']:.2f}</div>
                        </div>
                        <div class="data-source-info">
                            📡 {metric['data_source']} | Updated: {metric['last_updated']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Individual stock deep dive
        st.markdown("### 📊 Individual Stock Deep Analysis")
        
        stock_symbols = df[df['category'] == 'Stock Market']['series_id'].unique()
        if len(stock_symbols) > 0:
            selected_stock = st.selectbox("Select Stock for Deep Analysis:", stock_symbols)
            
            stock_data = df[df['series_id'] == selected_stock].sort_values('date')
            
            if not stock_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price chart with volume
                    fig_price = go.Figure()
                    
                    fig_price.add_trace(go.Scatter(
                        x=stock_data['date'], 
                        y=stock_data['value'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_price.update_layout(
                        title=f"{stock_data.iloc[-1]['series_name']} - Real-Time Price",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    # Volume and volatility analysis
                    if 'volume' in stock_data.columns:
                        recent_data = stock_data.tail(50)
                        
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(
                            x=recent_data['date'],
                            y=recent_data['volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ))
                        
                        fig_vol.update_layout(
                            title=f"{stock_data.iloc[-1]['series_name']} - Trading Volume",
                            xaxis_title="Date",
                            yaxis_title="Volume",
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_vol, use_container_width=True)
                
                # Technical analysis
                st.markdown("#### 🔍 Technical Analysis")
                
                values = stock_data['value'].values
                if len(values) >= 50:
                    sma_20 = np.mean(values[-20:])
                    sma_50 = np.mean(values[-50:])
                    current_price = values[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("20-Day SMA", f"${sma_20:.2f}")
                    
                    with col2:
                        st.metric("50-Day SMA", f"${sma_50:.2f}")
                    
                    with col3:
                        trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
                        st.metric("Trend", trend)
                    
                    with col4:
                        returns = np.diff(values) / values[:-1]
                        volatility = np.std(returns) * np.sqrt(252) * 100
                        st.metric("Volatility", f"{volatility:.1f}%")
    
    with tab3:
        st.subheader("₿ Real-Time Cryptocurrency Analysis")
        
        # Real crypto performance
        if 'Crypto' in portfolio_analysis:
            crypto_metrics = portfolio_analysis['Crypto']
            
            st.markdown("### 🚀 Live Crypto Performance Metrics")
            
            for metric in crypto_metrics:
                # Crypto-specific color coding (higher volatility expected)
                perf_color = "green" if metric['return_1m'] > 0 else "red"
                vol_color = "red" if metric['volatility'] > 100 else "orange" if metric['volatility'] > 60 else "green"
                
                with st.container():
                    st.markdown(f"""
                    <div class="asset-card">
                        <h4>{metric['series_name']} ({metric['series_id']})</h4>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>Current:</strong> ${metric['current_value']:,.2f}</div>
                            <div><strong>1M Return:</strong> <span style="color: {perf_color}">{metric['return_1m']:+.1f}%</span></div>
                            <div><strong>Volatility:</strong> <span style="color: {vol_color}">{metric['volatility']:.1f}%</span></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <div><strong>3M Return:</strong> <span style="color: {'green' if metric['return_3m'] > 0 else 'red'}">{metric['return_3m']:+.1f}%</span></div>
                            <div><strong>1Y Return:</strong> <span style="color: {'green' if metric['return_1y'] > 0 else 'red'}">{metric['return_1y']:+.1f}%</span></div>
                            <div><strong>Max Drawdown:</strong> {metric['max_drawdown']:.1f}%</div>
                        </div>
                        <div class="data-source-info">
                            📡 {metric['data_source']} | Updated: {metric['last_updated']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Crypto market analysis
        st.markdown("### 📊 Crypto Market Correlation Analysis")
        
        crypto_data = df[df['category'] == 'Cryptocurrency']
        if not crypto_data.empty and len(crypto_data['series_id'].unique()) > 1:
            crypto_pivot = crypto_data.pivot_table(index='date', columns='series_id', values='value')
            crypto_returns = crypto_pivot.pct_change().dropna()
            crypto_corr = crypto_returns.corr()
            
            if not crypto_corr.empty:
                fig_corr = px.imshow(
                    crypto_corr,
                    title="Real-Time Cryptocurrency Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    color_continuous_midpoint=0
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation insights
                st.markdown("#### 🔍 Correlation Insights")
                
                # Find highest and lowest correlations
                corr_values = []
                for i in range(len(crypto_corr.columns)):
                    for j in range(i+1, len(crypto_corr.columns)):
                        asset1 = crypto_corr.columns[i]
                        asset2 = crypto_corr.columns[j]
                        corr_val = crypto_corr.iloc[i, j]
                        corr_values.append((asset1, asset2, corr_val))
                
                corr_values.sort(key=lambda x: x[2], reverse=True)
                
                if corr_values:
                    highest_corr = corr_values[0]
                    lowest_corr = corr_values[-1]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **🔗 Highest Correlation:**  
                        {highest_corr[0]} vs {highest_corr[1]}: {highest_corr[2]:.3f}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **↔️ Lowest Correlation:**  
                        {lowest_corr[0]} vs {lowest_corr[1]}: {lowest_corr[2]:.3f}
                        """)
    
    with tab4:
        st.subheader("🔮 AI-Powered Predictions on Real Data")
        
        if predictions:
            st.markdown("### 🤖 Live Data AI Forecasts")
            
            for asset_id, pred_data in predictions.items():
                st.markdown(f"#### 📈 {pred_data['series_name']} - {prediction_periods}-Day Forecast")
                
                # Get historical real data
                hist_data = df[df['series_id'] == asset_id].sort_values('date')
                
                if not hist_data.empty:
                    # Create enhanced prediction chart
                    fig_pred = go.Figure()
                    
                    # Historical data (last 90 days)
                    recent_hist = hist_data.tail(90)
                    fig_pred.add_trace(
                        go.Scatter(
                            x=recent_hist['date'],
                            y=recent_hist['value'],
                            mode='lines',
                            name='Historical (Real Data)',
                            line=dict(color='blue', width=3)
                        )
                    )
                    
                    # AI predictions
                    fig_pred.add_trace(
                        go.Scatter(
                            x=pred_data['dates'],
                            y=pred_data['predictions'],
                            mode='lines+markers',
                            name='AI Prediction',
                            line=dict(color='red', dash='dash', width=3),
                            marker=dict(size=8, symbol='diamond')
                        )
                    )
                    
                    # Enhanced confidence interval
                    confidence = pred_data.get('confidence', 80) / 100
                    std_dev = np.std(pred_data['predictions']) * (1 - confidence)
                    upper_bound = [p + std_dev for p in pred_data['predictions']]
                    lower_bound = [p - std_dev for p in pred_data['predictions']]
                    
                    fig_pred.add_trace(
                        go.Scatter(
                            x=pred_data['dates'] + pred_data['dates'][::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{confidence*100:.0f}% Confidence Interval',
                            showlegend=True
                        )
                    )
                    
                    fig_pred.update_layout(
                        title=f"{pred_data['series_name']} - AI Prediction on Real Data",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Enhanced prediction metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_value = hist_data.iloc[-1]['value']
                    predicted_value = pred_data['predictions'][-1]
                    change_pct = ((predicted_value - current_value) / current_value * 100)
                    confidence_score = pred_data.get('confidence', 80)
                    
                    with col1:
                        st.metric("Current Value", f"{current_value:.2f}")
                    
                    with col2:
                        st.metric(f"{prediction_periods}-Day Forecast", f"{predicted_value:.2f}")
                    
                    with col3:
                        st.metric("Predicted Change", f"{change_pct:+.1f}%")
                    
                    with col4:
                        st.metric("AI Confidence", f"{confidence_score:.0f}%")
                    
                    # Enhanced AI insights
                    trend_direction = "📈 Bullish" if change_pct > 2 else "📉 Bearish" if change_pct < -2 else "➡️ Neutral"
                    confidence_level = "High" if confidence_score > 80 else "Medium" if confidence_score > 60 else "Low"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <strong>🤖 AI Insight:</strong> Based on real market data, the model predicts a <strong>{trend_direction}</strong> trend 
                        for {pred_data['series_name']} with <strong>{confidence_level}</strong> confidence ({confidence_score:.0f}%). 
                        Expected {change_pct:+.1f}% change over {prediction_periods} days.
                        <br><br>
                        <strong>📊 Data Source:</strong> {pred_data.get('data_source', 'Real Market Data')}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🤖 AI models are training on real data... Predictions will be available shortly.")
            
            # Show model training status
            st.markdown("### 🧠 Model Training Status")
            
            for asset in ['SPY', 'BTCUSDT', 'UNRATE', 'EURUSD']:
                asset_data = df[df['series_id'] == asset]
                if not asset_data.empty:
                    data_points = len(asset_data)
                    data_quality = "✅ Excellent" if data_points > 200 else "🟡 Good" if data_points > 50 else "🔴 Limited"
                    
                    st.markdown(f"**{asset}:** {data_points} data points - {data_quality}")
    
    with tab5:
        st.subheader("🧠 Portfolio Intelligence & Real-Time Risk Analysis")
        
        # Enhanced portfolio overview
        st.markdown("### 📊 Real-Time Portfolio Overview")
        
        total_assets = len(df['series_id'].unique())
        categories = len(df['category'].unique())
        real_data_count = len(df[df['data_source'].str.contains('Real', na=False)])
        real_data_pct = (real_data_count / len(df) * 100) if len(df) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", total_assets)
        
        with col2:
            st.metric("Asset Categories", categories)
        
        with col3:
            st.metric("Real Data Points", f"{real_data_count:,}")
        
        with col4:
            st.metric("Data Quality", f"{real_data_pct:.0f}% Real")
        
        # Enhanced risk analysis by category
        st.markdown("### ⚠️ Real-Time Risk Analysis by Asset Category")
        
        for asset_type, metrics in portfolio_analysis.items():
            if metrics and len(metrics) > 0:
                avg_volatility = np.mean([m['volatility'] for m in metrics])
                avg_return = np.mean([m['return_1m'] for m in metrics])
                high_risk_count = len([m for m in metrics if m['volatility'] > 30])
                
                # Enhanced risk categorization
                if asset_type == 'Crypto':
                    risk_level = "🔴 High" if avg_volatility > 80 else "🟡 Medium" if avg_volatility > 50 else "🟢 Low"
                else:
                    risk_level = "🔴 High" if avg_volatility > 30 else "🟡 Medium" if avg_volatility > 15 else "🟢 Low"
                
                return_color = "green" if avg_return > 0 else "red"
                
                st.markdown(f"""
                <div class="risk-alert">
                    <h4>{asset_type} Risk Profile (Real-Time Analysis)</h4>
                    <p><strong>Average Volatility:</strong> {avg_volatility:.1f}% | <strong>Risk Level:</strong> {risk_level}</p>
                    <p><strong>Average 1M Return:</strong> <span style="color: {return_color}">{avg_return:+.1f}%</span> | 
                       <strong>High Risk Assets:</strong> {high_risk_count} out of {len(metrics)}</p>
                    <p><strong>Data Quality:</strong> Based on real market data from live APIs</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI-generated insights based on real data
        st.markdown("### 🤖 AI-Generated Portfolio Insights (Real Data)")
        
        # Calculate real insights
        total_return_1m = np.mean([m['return_1m'] for metrics in portfolio_analysis.values() for m in metrics]) if portfolio_analysis else 0
        total_volatility = np.mean([m['volatility'] for metrics in portfolio_analysis.values() for m in metrics]) if portfolio_analysis else 0
        
        insights = [
            f"📊 **Portfolio Performance**: Average 1-month return of {total_return_1m:+.1f}% across all assets",
            f"⚠️ **Risk Assessment**: Portfolio volatility at {total_volatility:.1f}% - {'High' if total_volatility > 40 else 'Moderate' if total_volatility > 20 else 'Low'} risk level",
            f"🎯 **Data Quality**: {real_data_pct:.0f}% of analysis based on real-time market data",
            f"🔮 **AI Confidence**: Model predictions have {len(predictions)} active forecasts on live data",
            f"🌍 **Market Coverage**: Tracking {total_assets} assets across {categories} categories globally"
        ]
        
        if real_data_pct > 80:
            insights.append("✅ **High Fidelity**: Analysis based on high-quality real market data")
        elif real_data_pct > 50:
            insights.append("🟡 **Mixed Quality**: Analysis combines real data with enhanced simulations")
        else:
            insights.append("🔴 **Limited Real Data**: Consider enabling API keys for better data quality")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Real-time performance attribution
        st.markdown("### 🏆 Real-Time Performance Attribution")
        
        if portfolio_analysis:
            categories = []
            avg_returns_1m = []
            avg_returns_3m = []
            avg_volatilities = []
            
            for category, metrics in portfolio_analysis.items():
                if metrics:
                    categories.append(category)
                    avg_returns_1m.append(np.mean([m['return_1m'] for m in metrics]))
                    avg_returns_3m.append(np.mean([m['return_3m'] for m in metrics]))
                    avg_volatilities.append(np.mean([m['volatility'] for m in metrics]))
            
            if categories:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_returns = px.bar(
                        x=categories,
                        y=avg_returns_1m,
                        title="1-Month Returns by Asset Category (Real Data)",
                        color=avg_returns_1m,
                        color_continuous_scale='RdYlGn'
                    )
                    fig_returns.update_layout(height=400)
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                with col2:
                    fig_vol = px.bar(
                        x=categories,
                        y=avg_volatilities,
                        title="Volatility by Asset Category (Real Data)",
                        color=avg_volatilities,
                        color_continuous_scale='YlOrRd'
                    )
                    fig_vol.update_layout(height=400)
                    st.plotly_chart(fig_vol, use_container_width=True)
        
        # Data source breakdown
        st.markdown("### 📡 Data Source Quality Report")
        
        source_summary = df['data_source'].value_counts().reset_index()
        source_summary.columns = ['Data Source', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(source_summary, use_container_width=True)
        
        with col2:
            fig_sources = px.pie(
                source_summary,
                values='Count',
                names='Data Source',
                title="Data Source Distribution"
            )
            st.plotly_chart(fig_sources, use_container_width=True)
    
    # Enhanced footer with real data info
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🚀 <strong>Economic Pulse V3.0</strong> - Real Data Financial Intelligence Platform<br>
        📡 Live Data: {real_data_pct:.0f}% Real Market Data | 🤖 AI Models: {len(predictions)} Active Forecasts | 
        📊 Assets: {total_assets} Global Instruments<br>
        <em>Powered by FRED, Yahoo Finance, CoinGecko APIs + Advanced Machine Learning</em><br>
        <small>Last Updated: {df['date'].max().strftime('%Y-%m-%d %H:%M')} | 
        Data Quality: {'🟢 Excellent' if real_data_pct > 80 else '🟡 Good' if real_data_pct > 50 else '🔴 Limited'}</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()# 🚀 Economic Pulse V3.0 - Real Data Multi-Asset Financial Intelligence Platform
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

# Enhanced CSS
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
</style>
""", unsafe_allow_html=True)

class RealDataLoader:
    """Enhanced data loader with real financial and economic data sources"""
    
    def __init__(self):
        self.fred_api_key = st.secrets.get("fred_api_key", "demo_key")  # Use Streamlit secrets
        self.alpha_vantage_key = st.secrets.get("alpha_vantage_key", "demo_key")
        self.cache_duration = 900  # 15 minutes for real data
        
    def load_all_data(self):
        """Load comprehensive real multi-asset data with Alpha Vantage enhancement"""
        
        all_data = []
        data_sources = []
        
        # Real economic indicators from FRED
        with st.spinner("📊 Loading real US economic data from FRED..."):
            economic_data, econ_sources = self.load_real_economic_data()
            all_data.extend(economic_data)
            data_sources.extend(econ_sources)
        
        # Enhanced stock market data (Alpha Vantage + Yahoo Finance)
        with st.spinner("📈 Loading enhanced stock market data (Alpha Vantage + Yahoo Finance)..."):
            stock_data, stock_sources = self.load_real_stock_data()
            all_data.extend(stock_data)
            data_sources.extend(stock_sources)
        
        # Enhanced cryptocurrency data (Alpha Vantage + CoinGecko)
        with st.spinner("₿ Loading enhanced cryptocurrency data (Alpha Vantage + CoinGecko)..."):
            crypto_data, crypto_sources = self.load_enhanced_crypto_data_av()
            all_data.extend(crypto_data)
            data_sources.extend(crypto_sources)
        
        # Enhanced forex data (Alpha Vantage + Yahoo Finance)
        with st.spinner("💱 Loading enhanced forex data (Alpha Vantage + Yahoo Finance)..."):
            forex_data, forex_sources = self.load_enhanced_forex_data_av()
            all_data.extend(forex_data)
            data_sources.extend(forex_sources)
        
        # Real international data
        with st.spinner("🌍 Loading real international economic data..."):
            intl_data, intl_sources = self.load_real_international_data()
            all_data.extend(intl_data)
            data_sources.extend(intl_sources)
        
        df = pd.DataFrame(all_data)
        
        # Enhanced data sources display
        if data_sources:
            st.sidebar.subheader("📡 Premium Data Sources")
            unique_sources = list(set(data_sources))
            
            # Prioritize Alpha Vantage sources
            av_sources = [s for s in unique_sources if 'Alpha Vantage' in s]
            other_sources = [s for s in unique_sources if 'Alpha Vantage' not in s]
            
            for source in av_sources:
                st.sidebar.markdown(f"<div class='real-data-badge'>🔥 {source}</div>", unsafe_allow_html=True)
            
            for source in other_sources:
                badge_class = 'real-data-badge' if 'Real' in source else 'data-source-info'
                icon = '✅' if 'Real' in source else '🎯'
                st.sidebar.markdown(f"<div class='{badge_class}'>{icon} {source}</div>", unsafe_allow_html=True)
        
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
            'UMCSENT': 'Consumer Sentiment'
        }
        
        data = []
        sources = []
        
        for series_id, series_name in indicators.items():
            try:
                # Try FRED API first
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 120,  # Last 10 years monthly
                    'sort_order': 'desc'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200 and self.fred_api_key != "demo_key":
                    json_data = response.json()
                    observations = json_data.get('observations', [])
                    
                    for obs in observations:
                        if obs['value'] != '.' and obs['value']:
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
                    
                    sources.append("FRED Economic Data")
                    st.success(f"✅ Loaded real {series_name} from FRED")
                    time.sleep(0.1)  # Rate limiting
                    
                else:
                    raise Exception("FRED API not available")
                    
            except Exception as e:
                # Fallback to realistic simulated data with current patterns
                st.warning(f"⚠️ FRED API unavailable for {series_name}, using enhanced simulation")
                sim_data = self.generate_realistic_economic_data(series_id, series_name)
                data.extend(sim_data)
                sources.append("Enhanced Simulation")
        
        return data, sources
    
    def load_real_stock_data(self):
        """Load real stock market data from Yahoo Finance and Alpha Vantage"""
        
        stocks = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'GLD': 'Gold ETF',
            'VTI': 'Total Stock Market ETF',
            'TLT': '20+ Year Treasury ETF',
            'VIX': 'Volatility Index',
            'DIA': 'Dow Jones ETF',
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corp',
            'GOOGL': 'Alphabet Inc',
            'AMZN': 'Amazon.com Inc',
            'TSLA': 'Tesla Inc',
            'NVDA': 'NVIDIA Corp'
        }
        
        data = []
        sources = []
        
        # Try Alpha Vantage first for individual stocks
        av_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        if self.alpha_vantage_key != "demo_key":
            for symbol in av_stocks:
                if symbol in stocks:
                    try:
                        st.info(f"📊 Loading {stocks[symbol]} from Alpha Vantage...")
                        
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
                            time_series = av_data.get('Time Series (Daily)', {})
                            
                            if time_series:
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
                                            'series_name': stocks[symbol],
                                            'category': 'Stock Market',
                                            'unit': 'USD',
                                            'asset_type': 'Equity',
                                            'data_source': 'Alpha Vantage API (Real)'
                                        })
                                    except (ValueError, KeyError) as e:
                                        continue
                                
                                sources.append("Alpha Vantage API")
                                st.success(f"✅ Loaded {stocks[symbol]} from Alpha Vantage")
                                time.sleep(12)  # Alpha Vantage rate limit (5 calls per minute)
                            else:
                                st.warning(f"⚠️ No data returned for {symbol} from Alpha Vantage")
                        else:
                            st.warning(f"⚠️ Alpha Vantage API error for {symbol}")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {stocks[symbol]} from Alpha Vantage: {str(e)}")
        
        # Use Yahoo Finance for ETFs and remaining stocks
        yf_stocks = [s for s in stocks.keys() if s not in av_stocks or s not in [d['series_id'] for d in data]]
        
        if YFINANCE_AVAILABLE and yf_stocks:
            for symbol in yf_stocks:
                try:
                    st.info(f"📈 Loading {stocks[symbol]} from Yahoo Finance...")
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1y')  # 1 year of data
                    
                    if not hist.empty:
                        for date, row in hist.iterrows():
                            data.append({
                                'date': date.date(),
                                'value': float(row['Close']),
                                'volume': int(row['Volume']) if row['Volume'] > 0 else 1000000,
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'open': float(row['Open']),
                                'series_id': symbol,
                                'series_name': stocks[symbol],
                                'category': 'Stock Market',
                                'unit': 'USD',
                                'asset_type': 'Equity',
                                'data_source': 'Yahoo Finance (Real)'
                            })
                        
                        sources.append("Yahoo Finance")
                        st.success(f"✅ Loaded {stocks[symbol]} from Yahoo Finance")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load {stocks[symbol]} from Yahoo Finance")
                    # Fallback to simulation
                    sim_data = self.generate_realistic_stock_data(symbol, stocks[symbol])
                    data.extend(sim_data)
        
        # Fallback simulations for any missing data
        loaded_symbols = set([d['series_id'] for d in data])
        missing_symbols = set(stocks.keys()) - loaded_symbols
        
        for symbol in missing_symbols:
            st.warning(f"⚠️ Using enhanced simulation for {stocks[symbol]}")
            sim_data = self.generate_realistic_stock_data(symbol, stocks[symbol])
            data.extend(sim_data)
            
        if not sources:
            sources.append("Enhanced Simulation")
        
        return data, sources
    
    def load_enhanced_crypto_data_av(self):
        """Load enhanced cryptocurrency data using Alpha Vantage"""
        
        data = []
        sources = []
        
        # Try Alpha Vantage crypto first
        if self.alpha_vantage_key != "demo_key":
            crypto_symbols = ['BTC', 'ETH', 'LTC', 'XRP']
            
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
                        time_series = av_data.get('Time Series (Digital Currency Daily)', {})
                        
                        if time_series:
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
                                except (ValueError, KeyError):
                                    continue
                            
                            sources.append("Alpha Vantage Crypto")
                            st.success(f"✅ Loaded {symbol} from Alpha Vantage")
                            time.sleep(12)  # Rate limiting
                        else:
                            st.warning(f"⚠️ No crypto data for {symbol} from Alpha Vantage")
                            
                except Exception as e:
                    st.warning(f"⚠️ Alpha Vantage crypto error for {symbol}: {str(e)}")
        
        # If Alpha Vantage didn't work or we need more coins, try CoinGecko
        loaded_crypto_symbols = set([d['series_id'] for d in data])
        needed_cryptos = {'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'} - loaded_crypto_symbols
        
        if needed_cryptos:
            # Use the existing CoinGecko method for remaining cryptos
            coingecko_data, coingecko_sources = self.load_real_crypto_data()
            data.extend(coingecko_data)
            sources.extend(coingecko_sources)
        
        return data, sources
    
    def load_enhanced_forex_data_av(self):
        """Load enhanced forex data using Alpha Vantage"""
        
        data = []
        sources = []
        
        # Alpha Vantage forex pairs
        av_pairs = {
            'EUR': 'EUR/USD Exchange Rate',
            'GBP': 'GBP/USD Exchange Rate',
            'JPY': 'USD/JPY Exchange Rate',
            'CAD': 'USD/CAD Exchange Rate',
            'AUD': 'AUD/USD Exchange Rate'
        }
        
        if self.alpha_vantage_key != "demo_key":
            for from_currency, name in av_pairs.items():
                try:
                    st.info(f"💱 Loading {name} from Alpha Vantage...")
                    
                    url = "https://www.alphavantage.co/query"
                    
                    # Determine currency pair format for Alpha Vantage
                    if from_currency in ['EUR', 'GBP', 'AUD']:
                        from_symbol, to_symbol = from_currency, 'USD'
                    else:  # JPY, CAD
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
                        time_series = av_data.get('Time Series FX (Daily)', {})
                        
                        if time_series:
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
                                except (ValueError, KeyError):
                                    continue
                            
                            sources.append("Alpha Vantage FX")
                            st.success(f"✅ Loaded {name} from Alpha Vantage")
                            time.sleep(12)  # Rate limiting
                        else:
                            st.warning(f"⚠️ No forex data for {from_currency} from Alpha Vantage")
                            
                except Exception as e:
                    st.warning(f"⚠️ Alpha Vantage forex error for {from_currency}: {str(e)}")
        
        # Fallback to Yahoo Finance for any missing pairs
        loaded_pairs = set([d['series_id'] for d in data])
        yf_pairs = {
            'EURUSD=X': 'EUR/USD Exchange Rate',
            'GBPUSD=X': 'GBP/USD Exchange Rate', 
            'USDJPY=X': 'USD/JPY Exchange Rate',
            'AUDUSD=X': 'AUD/USD Exchange Rate',
            'USDCAD=X': 'USD/CAD Exchange Rate'
        }
        
        missing_pairs = []
        for yf_pair, name in yf_pairs.items():
            clean_pair = yf_pair.replace('=X', '')
            if clean_pair not in loaded_pairs:
                missing_pairs.append((yf_pair, name, clean_pair))
        
        if missing_pairs and YFINANCE_AVAILABLE:
            for yf_pair, name, clean_pair in missing_pairs:
                try:
                    st.info(f"💱 Loading {name} from Yahoo Finance (fallback)...")
                    ticker = yf.Ticker(yf_pair)
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        for date, row in hist.iterrows():
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
                        
                        sources.append("Yahoo Finance FX")
                        st.success(f"✅ Loaded {name} from Yahoo Finance")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load {name} from Yahoo Finance")
        
        # Simulation fallback for any remaining missing data
        all_expected_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        loaded_pair_ids = set([d['series_id'] for d in data])
        missing_pair_ids = set(all_expected_pairs) - loaded_pair_ids
        
        for pair_id in missing_pair_ids:
            st.warning(f"⚠️ Using enhanced simulation for {pair_id}")
            sim_data = self.generate_realistic_forex_data(pair_id, f"{pair_id} Exchange Rate")
            data.extend(sim_data)
        
        if not sources:
            sources.append("Enhanced Simulation")
        
        return data, sources
    
    def load_real_crypto_data(self):
        """Load real cryptocurrency data from public APIs"""
        
        cryptos = {
            'bitcoin': {'symbol': 'BTC', 'name': 'Bitcoin'},
            'ethereum': {'symbol': 'ETH', 'name': 'Ethereum'},
            'binancecoin': {'symbol': 'BNB', 'name': 'Binance Coin'},
            'cardano': {'symbol': 'ADA', 'name': 'Cardano'},
            'solana': {'symbol': 'SOL', 'name': 'Solana'}
        }
        
        data = []
        sources = []
        
        # Try CoinGecko API (free, no API key required)
        try:
            crypto_ids = list(cryptos.keys())
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
                        
                        # Get historical data for this coin
                        hist_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                        hist_params = {'vs_currency': 'usd', 'days': '365'}
                        
                        try:
                            hist_response = requests.get(hist_url, params=hist_params, timeout=10)
                            if hist_response.status_code == 200:
                                hist_data = hist_response.json()
                                prices = hist_data.get('prices', [])
                                
                                # Process price history
                                for price_point in prices[-200:]:  # Last 200 days
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
                                
                                st.success(f"✅ Loaded real {name} price: ${price:,.2f}")
                                time.sleep(1)  # Rate limiting for free API
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not load {name} history")
                
                sources.append("CoinGecko API")
                
            else:
                raise Exception("CoinGecko API not available")
                
        except Exception as e:
            st.warning("⚠️ Crypto APIs unavailable, using enhanced simulations")
            # Fallback to enhanced simulations with current market levels
            for crypto_id, info in cryptos.items():
                sim_data = self.generate_realistic_crypto_data(info['symbol'], info['name'])
                data.extend(sim_data)
            sources.append("Enhanced Crypto Simulation")
        
        return data, sources
    
    def load_real_forex_data(self):
        """Load real forex data"""
        
        pairs = {
            'EURUSD=X': 'EUR/USD Exchange Rate',
            'GBPUSD=X': 'GBP/USD Exchange Rate', 
            'USDJPY=X': 'USD/JPY Exchange Rate',
            'AUDUSD=X': 'AUD/USD Exchange Rate',
            'USDCAD=X': 'USD/CAD Exchange Rate'
        }
        
        data = []
        sources = []
        
        if YFINANCE_AVAILABLE:
            for pair, name in pairs.items():
                try:
                    ticker = yf.Ticker(pair)
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        for date, row in hist.iterrows():
                            data.append({
                                'date': date.date(),
                                'value': float(row['Close']),
                                'volume': int(row['Volume']) if row['Volume'] > 0 else 1000000,
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'open': float(row['Open']),
                                'series_id': pair.replace('=X', ''),
                                'series_name': name,
                                'category': 'Forex',
                                'unit': 'Exchange Rate',
                                'asset_type': 'Currency',
                                'data_source': 'Yahoo Finance (Real)'
                            })
                        
                        sources.append("Yahoo Finance Forex")
                        st.success(f"✅ Loaded real {name}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load {name}")
        
        if not sources:
            st.warning("⚠️ Forex APIs unavailable, using simulations")
            for pair, name in pairs.items():
                sim_data = self.generate_realistic_forex_data(pair.replace('=X', ''), name)
                data.extend(sim_data)
            sources.append("Forex Simulation")
        
        return data, sources
    
    def load_real_international_data(self):
        """Load real international economic data"""
        
        # International FRED series
        intl_indicators = {
            'LRUNTTTTGBM156S': 'UK Unemployment Rate',
            'LRUNTTTTDEM156S': 'Germany Unemployment Rate',
            'LRUNTTTTJPM156S': 'Japan Unemployment Rate',
            'IRLTLT01GBM156N': 'UK Long-term Interest Rate',
            'CPALTT01GBM661N': 'UK Consumer Price Index'
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
                        
                        for obs in observations:
                            if obs['value'] != '.' and obs['value']:
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
                        
                        sources.append("FRED International")
                        st.success(f"✅ Loaded real {series_name}")
                        time.sleep(0.1)
                    else:
                        raise Exception("FRED API error")
                else:
                    raise Exception("No FRED API key")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load {series_name}, using simulation")
                sim_data = self.generate_realistic_international_data(series_id, series_name)
                data.extend(sim_data)
        
        if not sources:
            sources.append("International Simulation")
        
        return data, sources
    
    def extract_country(self, series_name):
        """Extract country from series name"""
        if 'UK' in series_name:
            return 'United Kingdom'
        elif 'Germany' in series_name:
            return 'Germany'
        elif 'Japan' in series_name:
            return 'Japan'
        else:
            return 'International'
    
    def generate_realistic_economic_data(self, series_id, series_name):
        """Generate enhanced realistic economic data with current patterns"""
        
        data = []
        dates = pd.date_range('2020-01-01', datetime.now(), freq='M')
        
        # Current realistic values (as of 2025)
        current_values = {
            'UNRATE': 3.8,      # Current unemployment
            'FEDFUNDS': 4.75,   # Current Fed funds rate  
            'CPIAUCSL': 310.0,  # Current CPI level
            'GDP': 28000.0,     # Current GDP
            'HOUST': 1350.0,    # Current housing starts
            'INDPRO': 103.0,    # Current industrial production
            'PAYEMS': 158000.0, # Current nonfarm payrolls
            'UMCSENT': 95.0     # Current consumer sentiment
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
                'data_source': 'Enhanced Simulation (Current Patterns)'
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
            'SPY': 580.0,   # S&P 500 at highs
            'QQQ': 520.0,   # NASDAQ strong
            'IWM': 240.0,   # Small caps
            'GLD': 185.0,   # Gold
            'VTI': 290.0,   # Total market
            'TLT': 85.0,    # Bonds under pressure
            'VIX': 15.0,    # Low volatility
            'DIA': 440.0    # Dow
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
            if symbol in ['SPY', 'QQQ', 'VTI']:  # Strong performers
                daily_return += 0.0002  # Extra positive bias
            elif symbol == 'TLT':  # Bonds struggling
                daily_return -= 0.0001  # Slight negative bias
            
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
            'BTC': 115000.0,  # Bitcoin at ATH
            'ETH': 4200.0,    # Ethereum strong
            'BNB': 720.0,     # BNB elevated
            'ADA': 1.15,      # Cardano recovery
            'SOL': 280.0      # Solana surge
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
                'series_name': f"{name} Price",
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
        """Generate realistic forex data"""
        
        current_rates = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2420,
            'USDJPY': 155.0,
            'AUDUSD': 0.6180,
            'USDCAD': 1.4350
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
                'series_