# 💼 Advanced Portfolio Optimization & Risk Management
# Modern Portfolio Theory, Risk Parity, and Advanced Optimization

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
try:
    import scipy.optimize as sco
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ SciPy not available. Install with: pip install scipy")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class ModernPortfolioOptimizer:
    """Advanced portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self):
        self.returns_data = None
        self.correlation_matrix = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.risk_free_rate = 0.02  # 2% default risk-free rate
        
    def prepare_returns_data(self, df, assets=None, period_days=252):
        """Prepare returns data for portfolio optimization"""
        
        if assets is None:
            # Get all equity assets
            equity_data = df[df['asset_type'] == 'Equity']
            assets = equity_data['series_id'].unique()[:10]  # Limit to 10 assets
        
        returns_dict = {}
        
        for asset in assets:
            asset_data = df[df['series_id'] == asset].sort_values('date')
            if len(asset_data) >= period_days:
                prices = asset_data['value'].values
                returns = np.diff(prices) / prices[:-1]
                returns_dict[asset] = returns[-period_days:]  # Last year of data
        
        if len(returns_dict) < 2:
            st.warning("❌ Insufficient data for portfolio optimization")
            return False
        
        # Create returns DataFrame
        min_length = min(len(returns) for returns in returns_dict.values())
        aligned_returns = {asset: returns[-min_length:] for asset, returns in returns_dict.items()}
        
        self.returns_data = pd.DataFrame(aligned_returns)
        
        # Calculate expected returns (annualized)
        self.expected_returns = self.returns_data.mean() * 252
        
        # Calculate covariance matrix (annualized)
        self.covariance_matrix = self.returns_data.cov() * 252
        
        # Calculate correlation matrix
        self.correlation_matrix = self.returns_data.corr()
        
        return True
    
    def calculate_portfolio_performance(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        
        if self.returns_data is None:
            return None, None, None
        
        weights = np.array(weights)
        
        # Expected return
        portfolio_return = np.sum(weights * self.expected_returns)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(self, optimization_type='max_sharpe'):
        """Optimize portfolio using different strategies"""
        
        if not SCIPY_AVAILABLE:
            st.error("🚫 SciPy required for portfolio optimization")
            return None
        
        if self.returns_data is None:
            st.error("❌ No returns data available")
            return None
        
        num_assets = len(self.returns_data.columns)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only positions
        
        # Initial guess (equal weights)
        initial_guess = np.array([1.0 / num_assets] * num_assets)
        
        if optimization_type == 'max_sharpe':
            # Maximize Sharpe ratio
            def objective(weights):
                _, _, sharpe = self.calculate_portfolio_performance(weights)
                return -sharpe  # Minimize negative Sharpe ratio
            
        elif optimization_type == 'min_volatility':
            # Minimize volatility
            def objective(weights):
                _, volatility, _ = self.calculate_portfolio_performance(weights)
                return volatility
            
        elif optimization_type == 'max_return':
            # Maximize return (subject to volatility constraint)
            def objective(weights):
                portfolio_return, _, _ = self.calculate_portfolio_performance(weights)
                return -portfolio_return
            
            # Add volatility constraint (max 20% volatility)
            volatility_constraint = {'type': 'ineq', 'fun': lambda x: 0.20 - self.calculate_portfolio_performance(x)[1]}
            constraints = [constraints, volatility_constraint]
        
        elif optimization_type == 'risk_parity':
            # Risk parity optimization
            def objective(weights):
                # Risk parity: each asset contributes equally to portfolio risk
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_volatility
                contrib = weights * marginal_contrib
                target_contrib = portfolio_volatility / num_assets
                return np.sum((contrib - target_contrib) ** 2)
        
        else:
            st.error(f"❌ Unknown optimization type: {optimization_type}")
            return None
        
        try:
            # Optimize
            result = sco.minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'assets': list(self.returns_data.columns),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_type': optimization_type,
                    'success': True
                }
            else:
                st.error(f"❌ Optimization failed: {result.message}")
                return None
                
        except Exception as e:
            st.error(f"❌ Optimization error: {str(e)}")
            return None
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """Generate efficient frontier for visualization"""
        
        if not SCIPY_AVAILABLE or self.returns_data is None:
            return None
        
        num_assets = len(self.returns_data.columns)
        results = []
        
        # Define return targets
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        for target_return in target_returns:
            # Minimize volatility for given return
            def objective(weights):
                _, volatility, _ = self.calculate_portfolio_performance(weights)
                return volatility
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}  # Target return
            ]
            
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = np.array([1.0 / num_assets] * num_assets)
            
            try:
                result = sco.minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    weights = result.x
                    portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(weights)
                    
                    results.append({
                        'return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': weights
                    })
            except:
                continue
        
        return pd.DataFrame(results) if results else None

class RiskManagementTools:
    """Advanced risk management and analysis tools"""
    
    def __init__(self):
        self.portfolio_data = None
    
    def calculate_var_cvar(self, returns, confidence_level=0.05):
        """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)"""
        
        if len(returns) == 0:
            return None, None
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # VaR calculation
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # CVaR calculation (expected shortfall)
        cvar = sorted_returns[:var_index].mean() if var_index > 0 else var
        
        return var, cvar
    
    def calculate_maximum_drawdown(self, prices):
        """Calculate maximum drawdown"""
        
        if len(prices) == 0:
            return 0, 0, 0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        cumulative = np.insert(cumulative, 0, 1)  # Start with 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        max_dd_index = np.argmin(drawdown)
        
        # Find peak before max drawdown
        peak_index = np.argmax(running_max[:max_dd_index+1])
        
        return max_drawdown, peak_index, max_dd_index
    
    def calculate_portfolio_beta(self, portfolio_returns, market_returns):
        """Calculate portfolio beta relative to market"""
        
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
            return None
        
        # Calculate beta using linear regression
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return None
        
        beta = covariance / market_variance
        return beta
    
    def stress_test_portfolio(self, weights, scenarios):
        """Perform stress testing on portfolio"""
        
        stress_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            # Apply scenario shocks to returns
            stressed_returns = []
            
            for asset, weight in zip(self.expected_returns.index, weights):
                if asset in scenario_data:
                    shock = scenario_data[asset]
                    stressed_return = self.expected_returns[asset] * (1 + shock)
                else:
                    stressed_return = self.expected_returns[asset]
                
                stressed_returns.append(stressed_return * weight)
            
            portfolio_stressed_return = sum(stressed_returns)
            stress_results[scenario_name] = portfolio_stressed_return
        
        return stress_results

class PortfolioDashboard:
    """Interactive portfolio optimization dashboard"""
    
    def __init__(self):
        self.optimizer = ModernPortfolioOptimizer()
        self.risk_manager = RiskManagementTools()
    
    def create_optimization_interface(self, df):
        """Create interactive portfolio optimization interface"""
        
        st.subheader("💼 Portfolio Optimization & Risk Management")
        
        # Sidebar controls
        st.sidebar.subheader("🎛️ Portfolio Configuration")
        
        # Asset selection
        available_assets = df[df['asset_type'] == 'Equity']['series_id'].unique()
        selected_assets = st.sidebar.multiselect(
            "Select Assets for Portfolio:",
            available_assets,
            default=list(available_assets[:5]) if len(available_assets) >= 5 else list(available_assets)
        )
        
        if len(selected_assets) < 2:
            st.warning("⚠️ Please select at least 2 assets for optimization")
            return
        
        # Optimization parameters
        optimization_type = st.sidebar.selectbox(
            "Optimization Strategy:",
            ["max_sharpe", "min_volatility", "max_return", "risk_parity"],
            format_func=lambda x: {
                "max_sharpe": "🎯 Maximum Sharpe Ratio",
                "min_volatility": "🛡️ Minimum Volatility",
                "max_return": "📈 Maximum Return",
                "risk_parity": "⚖️ Risk Parity"
            }[x]
        )
        
        risk_free_rate = st.sidebar.slider(
            "Risk-Free Rate (%):",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        ) / 100
        
        self.optimizer.risk_free_rate = risk_free_rate
        
        # Prepare data
        if not self.optimizer.prepare_returns_data(df, selected_assets):
            return
        
        # Optimization
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🚀 Optimize Portfolio", type="primary"):
                with st.spinner("🔄 Optimizing portfolio..."):
                    optimization_result = self.optimizer.optimize_portfolio(optimization_type)
                    
                    if optimization_result:
                        st.session_state['optimization_result'] = optimization_result
                        st.success("✅ Portfolio optimization completed!")
        
        # Display results
        if 'optimization_result' in st.session_state:
            self._display_optimization_results(st.session_state['optimization_result'])
        
        # Efficient frontier
        if st.checkbox("📊 Show Efficient Frontier"):
            self._display_efficient_frontier()
        
        # Risk analysis
        if st.checkbox("⚠️ Advanced Risk Analysis"):
            self._display_risk_analysis()
    
    def _display_optimization_results(self, result):
        """Display portfolio optimization results"""
        
        st.subheader("🎯 Optimal Portfolio Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Return",
                f"{result['expected_return']:.2%}",
                help="Annualized expected portfolio return"
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{result['volatility']:.2%}",
                help="Annualized portfolio volatility (risk)"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{result['sharpe_ratio']:.3f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            st.metric(
                "Strategy",
                result['optimization_type'].replace('_', ' ').title(),
                help="Optimization strategy used"
            )
        
        # Portfolio allocation
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart of allocations
            fig_pie = go.Figure(data=[go.Pie(
                labels=result['assets'],
                values=result['weights'],
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_pie.update_layout(
                title="🥧 Portfolio Allocation",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart of weights
            fig_bar = px.bar(
                x=result['assets'],
                y=result['weights'],
                title="📊 Asset Weights",
                labels={'x': 'Assets', 'y': 'Weight'},
                color=result['weights'],
                color_continuous_scale='viridis'
            )
            
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed allocation table
        st.subheader("📋 Detailed Allocation")
        
        allocation_df = pd.DataFrame({
            'Asset': result['assets'],
            'Weight': [f"{w:.2%}" for w in result['weights']],
            'Dollar Amount ($10,000)': [f"${w * 10000:.2f}" for w in result['weights']]
        })
        
        st.dataframe(allocation_df, use_container_width=True)
    
    def _display_efficient_frontier(self):
        """Display efficient frontier visualization"""
        
        st.subheader("📈 Efficient Frontier Analysis")
        
        with st.spinner("🔄 Generating efficient frontier..."):
            frontier_data = self.optimizer.generate_efficient_frontier()
        
        if frontier_data is not None and not frontier_data.empty:
            # Create efficient frontier plot
            fig = go.Figure()
            
            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=frontier_data['volatility'],
                y=frontier_data['return'],
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ))
            
            # Individual assets
            fig.add_trace(go.Scatter(
                x=np.sqrt(np.diag(self.optimizer.covariance_matrix)),
                y=self.optimizer.expected_returns,
                mode='markers',
                name='Individual Assets',
                marker=dict(size=10, color='red', symbol='diamond'),
                text=self.optimizer.expected_returns.index,
                textposition='top center'
            ))
            
            # Optimal portfolio (if available)
            if 'optimization_result' in st.session_state:
                result = st.session_state['optimization_result']
                fig.add_trace(go.Scatter(
                    x=[result['volatility']],
                    y=[result['expected_return']],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(size=15, color='green', symbol='star')
                ))
            
            fig.update_layout(
                title="📊 Efficient Frontier & Asset Allocation",
                xaxis_title="Volatility (Risk)",
                yaxis_title="Expected Return",
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk-return table
            st.subheader("📊 Risk-Return Analysis")
            
            summary_stats = pd.DataFrame({
                'Metric': ['Min Volatility', 'Max Return', 'Max Sharpe Ratio'],
                'Volatility': [
                    f"{frontier_data['volatility'].min():.2%}",
                    f"{frontier_data.loc[frontier_data['return'].idxmax(), 'volatility']:.2%}",
                    f"{frontier_data.loc[frontier_data['sharpe_ratio'].idxmax(), 'volatility']:.2%}"
                ],
                'Return': [
                    f"{frontier_data.loc[frontier_data['volatility'].idxmin(), 'return']:.2%}",
                    f"{frontier_data['return'].max():.2%}",
                    f"{frontier_data.loc[frontier_data['sharpe_ratio'].idxmax(), 'return']:.2%}"
                ],
                'Sharpe Ratio': [
                    f"{frontier_data.loc[frontier_data['volatility'].idxmin(), 'sharpe_ratio']:.3f}",
                    f"{frontier_data.loc[frontier_data['return'].idxmax(), 'sharpe_ratio']:.3f}",
                    f"{frontier_data['sharpe_ratio'].max():.3f}"
                ]
            })
            
            st.dataframe(summary_stats, use_container_width=True)
        else:
            st.error("❌ Could not generate efficient frontier")
    
    def _display_risk_analysis(self):
        """Display advanced risk analysis"""
        
        st.subheader("⚠️ Advanced Risk Analysis")
        
        if 'optimization_result' not in st.session_state:
            st.warning("⚠️ Please optimize a portfolio first")
            return
        
        result = st.session_state['optimization_result']
        
        # Calculate portfolio returns for risk analysis
        portfolio_returns = np.dot(self.optimizer.returns_data, result['weights'])
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # VaR and CVaR
            var_95, cvar_95 = self.risk_manager.calculate_var_cvar(portfolio_returns, 0.05)
            var_99, cvar_99 = self.risk_manager.calculate_var_cvar(portfolio_returns, 0.01)
            
            st.markdown("**📉 Value at Risk (VaR)**")
            st.metric("95% VaR (Daily)", f"{var_95:.2%}", help="Expected loss not exceeded 95% of the time")
            st.metric("99% VaR (Daily)", f"{var_99:.2%}", help="Expected loss not exceeded 99% of the time")
            st.metric("95% CVaR (Daily)", f"{cvar_95:.2%}", help="Expected loss given VaR is exceeded")
        
        with col2:
            # Correlation analysis
            st.markdown("**🔗 Correlation Analysis**")
            
            # Average correlation
            avg_correlation = self.optimizer.correlation_matrix.values[np.triu_indices_from(self.optimizer.correlation_matrix.values, 1)].mean()
            max_correlation = self.optimizer.correlation_matrix.values[np.triu_indices_from(self.optimizer.correlation_matrix.values, 1)].max()
            min_correlation = self.optimizer.correlation_matrix.values[np.triu_indices_from(self.optimizer.correlation_matrix.values, 1)].min()
            
            st.metric("Average Correlation", f"{avg_correlation:.3f}")
            st.metric("Max Correlation", f"{max_correlation:.3f}")
            st.metric("Min Correlation", f"{min_correlation:.3f}")
        
        with col3:
            # Portfolio statistics
            st.markdown("**📊 Portfolio Statistics**")
            
            portfolio_skew = stats.skew(portfolio_returns)
            portfolio_kurtosis = stats.kurtosis(portfolio_returns)
            downside_deviation = np.std(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252)
            
            st.metric("Skewness", f"{portfolio_skew:.3f}", help="Asymmetry of return distribution")
            st.metric("Kurtosis", f"{portfolio_kurtosis:.3f}", help="Tail heaviness of distribution")
            st.metric("Downside Deviation", f"{downside_deviation:.2%}", help="Volatility of negative returns")
        
        # Correlation heatmap
        st.subheader("🔥 Asset Correlation Matrix")
        
        fig_corr = px.imshow(
            self.optimizer.correlation_matrix,
            title="Asset Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto=True
        )
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Stress testing
        st.subheader("🧪 Stress Testing Scenarios")
        
        # Define stress scenarios
        stress_scenarios = {
            "Market Crash (-20%)": {asset: -0.20 for asset in result['assets']},
            "Sector Rotation": {asset: np.random.uniform(-0.15, 0.15) for asset in result['assets']},
            "High Volatility": {asset: -0.10 for asset in result['assets']},
            "Interest Rate Shock": {asset: np.random.uniform(-0.12, -0.05) for asset in result['assets']}
        }
        
        # Calculate stress test results
        self.risk_manager.expected_returns = self.optimizer.expected_returns
        stress_results = self.risk_manager.stress_test_portfolio(result['weights'], stress_scenarios)
        
        # Display stress test results
        stress_df = pd.DataFrame([
            {'Scenario': scenario, 'Portfolio Return': f"{return_val:.2%}"} 
            for scenario, return_val in stress_results.items()
        ])
        
        st.dataframe(stress_df, use_container_width=True)

# Portfolio rebalancing tools
class PortfolioRebalancer:
    """Tools for portfolio rebalancing and maintenance"""
    
    def __init__(self):
        self.rebalancing_threshold = 0.05  # 5% threshold
        
    def check_rebalancing_needs(self, current_weights, target_weights, threshold=None):
        """Check if portfolio needs rebalancing"""
        
        if threshold is None:
            threshold = self.rebalancing_threshold
        
        weight_differences = np.abs(np.array(current_weights) - np.array(target_weights))
        max_difference = np.max(weight_differences)
        
        needs_rebalancing = max_difference > threshold
        
        return {
            'needs_rebalancing': needs_rebalancing,
            'max_difference': max_difference,
            'threshold': threshold,
            'weight_differences': weight_differences
        }
    
    def calculate_rebalancing_trades(self, current_weights, target_weights, portfolio_value):
        """Calculate trades needed for rebalancing"""
        
        current_values = np.array(current_weights) * portfolio_value
        target_values = np.array(target_weights) * portfolio_value
        
        trade_amounts = target_values - current_values
        
        return {
            'trade_amounts': trade_amounts,
            'buy_assets': trade_amounts > 0,
            'sell_assets': trade_amounts < 0,
            'total_turnover': np.sum(np.abs(trade_amounts))
        }