#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ML-Enhanced Economic Pulse Dashboard
# Advanced Machine Learning and AI Features

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🤖 Loading ML-Enhanced Economic Pulse Features...")

class EconomicMLPredictor:
    """Advanced ML prediction system for economic indicators"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def prepare_features(self, df, target_series, lookback_periods=12):
        """Prepare ML features from economic time series data"""

        # Get target series data
        target_data = df[df['series_id'] == target_series].sort_values('date')
        if len(target_data) < lookback_periods + 5:
            return None, None

        # Create lagged features
        features = []
        targets = []

        values = target_data['value'].values
        dates = target_data['date'].values

        for i in range(lookback_periods, len(values)):
            # Use previous N periods as features
            feature_vector = []

            # Lagged values
            for lag in range(1, lookback_periods + 1):
                feature_vector.append(values[i - lag])

            # Moving averages
            feature_vector.append(np.mean(values[i-3:i]))  # 3-period MA
            feature_vector.append(np.mean(values[i-6:i]))  # 6-period MA
            feature_vector.append(np.mean(values[i-12:i])) # 12-period MA

            # Volatility measures
            feature_vector.append(np.std(values[i-6:i]))    # 6-period volatility
            feature_vector.append(np.std(values[i-12:i]))  # 12-period volatility

            # Trend features
            recent_trend = np.polyfit(range(6), values[i-6:i], 1)[0]
            feature_vector.append(recent_trend)

            # Seasonal features (month)
            month = pd.to_datetime(dates[i]).month
            feature_vector.extend([1 if month == m else 0 for m in range(1, 13)])

            features.append(feature_vector)
            targets.append(values[i])

        return np.array(features), np.array(targets)

    def train_models(self, df, target_series):
        """Train multiple ML models for economic forecasting"""

        print(f"🔧 Training ML models for {target_series}...")

        X, y = self.prepare_features(df, target_series)
        if X is None:
            print(f"❌ Insufficient data for {target_series}")
            return False

        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }

        best_model = None
        best_score = float('inf')
        model_scores = {}

        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate performance
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            model_scores[name] = {'mse': mse, 'mae': mae}

            if mse < best_score:
                best_score = mse
                best_model = (name, model)

        # Store best model
        self.models[target_series] = best_model[1]
        self.scalers[target_series] = scaler if best_model[0] == 'Linear Regression' else None

        # Store feature importance (for tree-based models)
        if hasattr(best_model[1], 'feature_importances_'):
            self.feature_importance[target_series] = best_model[1].feature_importances_

        print(f"✅ Best model for {target_series}: {best_model[0]} (MSE: {best_score:.4f})")
        return True

    def predict_future(self, df, target_series, periods=6):
        """Generate future predictions"""

        if target_series not in self.models:
            print(f"❌ No trained model for {target_series}")
            return None

        # Get recent data
        target_data = df[df['series_id'] == target_series].sort_values('date')
        recent_values = target_data['value'].values[-12:]  # Last 12 periods

        predictions = []
        current_values = recent_values.copy()

        model = self.models[target_series]
        scaler = self.scalers[target_series]

        for _ in range(periods):
            # Prepare features (same as training)
            feature_vector = []

            # Lagged values
            for lag in range(1, 13):
                if lag <= len(current_values):
                    feature_vector.append(current_values[-lag])
                else:
                    feature_vector.append(current_values[0])  # Fallback

            # Moving averages
            feature_vector.append(np.mean(current_values[-3:]))
            feature_vector.append(np.mean(current_values[-6:]))
            feature_vector.append(np.mean(current_values[-12:]))

            # Volatility
            feature_vector.append(np.std(current_values[-6:]))
            feature_vector.append(np.std(current_values[-12:]))

            # Trend
            trend = np.polyfit(range(6), current_values[-6:], 1)[0]
            feature_vector.append(trend)

            # Seasonal (assuming monthly data, use current month + forecast step)
            current_month = pd.to_datetime(target_data.iloc[-1]['date']).month
            future_month = ((current_month + len(predictions) - 1) % 12) + 1
            feature_vector.extend([1 if future_month == m else 0 for m in range(1, 13)])

            # Make prediction
            X_pred = np.array([feature_vector])
            if scaler:
                X_pred = scaler.transform(X_pred)

            pred = model.predict(X_pred)[0]
            predictions.append(pred)

            # Update current values for next prediction
            current_values = np.append(current_values[1:], pred)

        # Create prediction dates
        last_date = target_data.iloc[-1]['date']
        pred_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]

        return {
            'dates': pred_dates,
            'predictions': predictions,
            'series_name': target_data.iloc[-1]['series_name']
        }

class EconomicAnomalyDetector:
    """ML-based anomaly detection for economic indicators"""

    def __init__(self, contamination=0.1):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False

    def detect_anomalies(self, df):
        """Detect anomalies in economic data"""

        print("🔍 Detecting economic anomalies...")

        anomalies = []

        for series_id in df['series_id'].unique():
            series_data = df[df['series_id'] == series_id].sort_values('date')

            if len(series_data) < 10:
                continue

            # Prepare features for anomaly detection
            values = series_data['value'].values.reshape(-1, 1)

            # Fit isolation forest
            self.model.fit(values)

            # Predict anomalies (-1 = anomaly, 1 = normal)
            predictions = self.model.predict(values)

            # Get anomaly scores
            scores = self.model.decision_function(values)

            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]

            for idx in anomaly_indices:
                anomalies.append({
                    'series_id': series_id,
                    'series_name': series_data.iloc[idx]['series_name'],
                    'date': series_data.iloc[idx]['date'],
                    'value': series_data.iloc[idx]['value'],
                    'anomaly_score': scores[idx],
                    'severity': 'High' if scores[idx] < -0.1 else 'Medium'
                })

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

class EconomicSentimentAnalyzer:
    """AI-powered economic sentiment analysis"""

    def analyze_trends(self, df):
        """Analyze economic trends and sentiment"""

        print("📈 Analyzing economic sentiment...")

        sentiment_scores = {}

        for series_id in df['series_id'].unique():
            series_data = df[df['series_id'] == series_id].sort_values('date')

            if len(series_data) < 12:
                continue

            values = series_data['value'].values

            # Calculate trend metrics
            recent_trend = np.polyfit(range(min(12, len(values))), values[-12:], 1)[0]
            volatility = np.std(values[-12:])
            momentum = values[-1] - values[-6] if len(values) >= 6 else 0

            # Calculate sentiment score
            # Positive trend, low volatility, positive momentum = bullish
            trend_score = np.tanh(recent_trend / np.std(values)) * 100
            volatility_score = max(0, 100 - volatility / np.mean(values) * 100)
            momentum_score = np.tanh(momentum / np.std(values)) * 100

            overall_sentiment = (trend_score + volatility_score + momentum_score) / 3

            # Classify sentiment
            if overall_sentiment > 20:
                sentiment_label = 'Bullish'
                emoji = '📈'
            elif overall_sentiment > -20:
                sentiment_label = 'Neutral'
                emoji = '➡️'
            else:
                sentiment_label = 'Bearish'
                emoji = '📉'

            sentiment_scores[series_id] = {
                'series_name': series_data.iloc[-1]['series_name'],
                'sentiment_score': overall_sentiment,
                'sentiment_label': sentiment_label,
                'emoji': emoji,
                'trend_score': trend_score,
                'volatility_score': volatility_score,
                'momentum_score': momentum_score
            }

        return sentiment_scores

def create_ml_enhanced_dashboard(df):
    """Create ML-enhanced dashboard with predictions and insights"""

    print("🤖 Creating ML-Enhanced Economic Dashboard...")

    # Initialize ML components
    predictor = EconomicMLPredictor()
    anomaly_detector = EconomicAnomalyDetector()
    sentiment_analyzer = EconomicSentimentAnalyzer()

    # Train models for key indicators
    key_indicators = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL']
    predictions = {}

    for indicator in key_indicators:
        if indicator in df['series_id'].values:
            if predictor.train_models(df, indicator):
                pred_results = predictor.predict_future(df, indicator, periods=6)
                if pred_results:
                    predictions[indicator] = pred_results

    # Detect anomalies
    anomalies = anomaly_detector.detect_anomalies(df)

    # Analyze sentiment
    sentiment = sentiment_analyzer.analyze_trends(df)

    # Create enhanced visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Unemployment Rate with ML Predictions',
            'Fed Funds Rate with Forecasts', 
            'Economic Sentiment Analysis',
            'Anomaly Detection Results'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot 1: Unemployment with predictions
    if 'UNRATE' in predictions:
        # Historical data (sample)
        hist_dates = pd.date_range('2020-01-01', '2025-01-01', freq='M')
        hist_values = 4.0 + np.random.normal(0, 0.8, len(hist_dates))
        hist_values[10:20] = 8.0 + np.random.normal(0, 1.0, 10)  # COVID spike

        fig.add_trace(
            go.Scatter(x=hist_dates, y=hist_values, mode='lines', 
                       name='Historical', line=dict(color=colors[0])),
            row=1, col=1
        )

        # Predictions
        pred_data = predictions['UNRATE']
        fig.add_trace(
            go.Scatter(x=pred_data['dates'], y=pred_data['predictions'], 
                       mode='lines+markers', name='ML Forecast',
                       line=dict(color='red', dash='dash')),
            row=1, col=1
        )

    # Plot 2: Fed Funds Rate
    if 'FEDFUNDS' in predictions:
        hist_dates = pd.date_range('2020-01-01', '2025-01-01', freq='M')
        hist_values = np.concatenate([
            np.full(24, 0.5),
            np.linspace(0.5, 5.0, len(hist_dates)-24)
        ])

        fig.add_trace(
            go.Scatter(x=hist_dates, y=hist_values, mode='lines',
                       name='Historical', line=dict(color=colors[1])),
            row=1, col=2
        )

        pred_data = predictions['FEDFUNDS']
        fig.add_trace(
            go.Scatter(x=pred_data['dates'], y=pred_data['predictions'],
                       mode='lines+markers', name='ML Forecast',
                       line=dict(color='red', dash='dash')),
            row=1, col=2
        )

    # Plot 3: Sentiment Analysis
    if sentiment:
        indicators = list(sentiment.keys())[:5]  # Top 5
        sentiment_scores = [sentiment[ind]['sentiment_score'] for ind in indicators]
        indicator_names = [sentiment[ind]['series_name'] for ind in indicators]

        fig.add_trace(
            go.Bar(x=indicator_names, y=sentiment_scores, 
                   name='Sentiment Score',
                   marker_color=['green' if s > 0 else 'red' for s in sentiment_scores]),
            row=2, col=1
        )

    # Plot 4: Anomaly Detection
    if not anomalies.empty:
        # Show recent anomalies
        recent_anomalies = anomalies.tail(10)
        fig.add_trace(
            go.Scatter(x=recent_anomalies['date'], y=recent_anomalies['value'],
                       mode='markers', name='Anomalies',
                       marker=dict(color='red', size=10, symbol='x')),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        title_text="ML-Enhanced Economic Dashboard",
        showlegend=True,
        template='plotly_white'
    )

    return {
        'dashboard': fig,
        'predictions': predictions,
        'anomalies': anomalies,
        'sentiment': sentiment
    }

# Function to generate AI insights
def generate_ml_insights(predictions, anomalies, sentiment):
    """Generate ML-powered economic insights"""

    insights = []

    insights.append("🤖 **ML-Powered Economic Insights:**\n")

    # Prediction insights
    if predictions:
        insights.append("**📈 ML Forecasts:**")
        for indicator, pred_data in predictions.items():
            latest_pred = pred_data['predictions'][-1]
            current_trend = "increasing" if pred_data['predictions'][-1] > pred_data['predictions'][0] else "decreasing"
            insights.append(f"• {pred_data['series_name']}: Predicted to be {current_trend} (6-month forecast: {latest_pred:.2f})")

    # Anomaly insights
    if not anomalies.empty:
        insights.append(f"\n**🔍 Anomaly Detection:**")
        high_severity = anomalies[anomalies['severity'] == 'High']
        if not high_severity.empty:
            insights.append(f"• {len(high_severity)} high-severity anomalies detected")
            for _, anomaly in high_severity.head(3).iterrows():
                insights.append(f"  - {anomaly['series_name']} on {anomaly['date'].strftime('%Y-%m-%d')}")

    # Sentiment insights
    if sentiment:
        insights.append(f"\n**📊 Economic Sentiment:**")
        bullish_count = sum(1 for s in sentiment.values() if s['sentiment_label'] == 'Bullish')
        bearish_count = sum(1 for s in sentiment.values() if s['sentiment_label'] == 'Bearish')

        insights.append(f"• {bullish_count} indicators showing bullish sentiment")
        insights.append(f"• {bearish_count} indicators showing bearish sentiment")

        # Top sentiment indicators
        sorted_sentiment = sorted(sentiment.items(), key=lambda x: x[1]['sentiment_score'], reverse=True)
        insights.append(f"• Most bullish: {sorted_sentiment[0][1]['series_name']} {sorted_sentiment[0][1]['emoji']}")
        insights.append(f"• Most bearish: {sorted_sentiment[-1][1]['series_name']} {sorted_sentiment[-1][1]['emoji']}")

    insights.append(f"\n*ML Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(insights)

# Example usage and testing
print("🧪 Testing ML-Enhanced Features...")

# For demonstration, create sample data
# Calculate the length of the date range
date_range_length = len(pd.date_range('2020-01-01', '2024-12-01', freq='M'))

sample_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', '2024-12-01', freq='M'),
    'value': 4.0 + np.random.normal(0, 0.5, date_range_length),
    'series_id': ['UNRATE'] * date_range_length, # Repeat 'UNRATE' for all rows
    'series_name': ['Unemployment Rate'] * date_range_length, # Repeat 'Unemployment Rate'
    'category': ['Employment'] * date_range_length, # Repeat 'Employment'
    'unit': ['%'] * date_range_length # Repeat '%'
})

# Add COVID spike
sample_data.loc[10:15, 'value'] = 8.0 + np.random.normal(0, 1.0, 6)

print("\n✅ ML-Enhanced Economic Pulse Features Ready!")
print("Features include:")
print("🔮 Machine Learning Forecasting")
print("🔍 Anomaly Detection")  
print("📈 Sentiment Analysis")
print("🤖 AI-Powered Insights")
print("📊 Enhanced Visualizations")

# Added this ML/AI integration test
# Define pipeline_results for demonstration purposes, as it's a dependency
# In a real scenario, pipeline_results would come from a data processing pipeline
pipeline_results = {
    'success': True,
    'data': sample_data # Using the sample_data created above
}

if pipeline_results['success']:
    print("\n🤖 Running ML-Enhanced Analysis...")

    # Create ML dashboard
    ml_results = create_ml_enhanced_dashboard(pipeline_results['data'])

    # Show ML dashboard
    if ml_results['dashboard']:
        ml_results['dashboard'].show()

    # Generate and show ML insights
    ml_insights = generate_ml_insights(
        ml_results['predictions'], 
        ml_results['anomalies'], 
        ml_results['sentiment']
    )

    print("\n" + ml_insights)


# In[ ]:




