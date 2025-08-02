# 🧠 Advanced LSTM Neural Networks for Financial Time Series Prediction
# Deep Learning Models with TensorFlow/Keras

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    DEEP_LEARNING_AVAILABLE = True
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. Install with: pip install tensorflow")

# Traditional ML fallbacks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

class AdvancedLSTMPredictor:
    """Advanced LSTM neural networks for financial time series prediction"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.feature_scalers = {}
        
    def prepare_lstm_data(self, df, target_series, include_features=True):
        """Prepare data for LSTM training with advanced features"""
        
        # Get target series data
        target_data = df[df['series_id'] == target_series].sort_values('date')
        if len(target_data) < self.sequence_length + 20:
            return None, None, None, None
        
        # Extract target values
        target_values = target_data['value'].values
        
        # Create advanced features
        features = []
        if include_features:
            # Price-based features
            features.append(target_values)  # Raw prices
            
            # Technical indicators
            features.append(self._calculate_sma(target_values, 5))    # 5-day SMA
            features.append(self._calculate_sma(target_values, 20))   # 20-day SMA
            features.append(self._calculate_ema(target_values, 12))   # 12-day EMA
            features.append(self._calculate_rsi(target_values, 14))   # RSI
            features.append(self._calculate_macd(target_values))      # MACD
            features.append(self._calculate_bollinger_bands(target_values)) # Bollinger Bands
            
            # Volatility features
            features.append(self._calculate_volatility(target_values, 10))  # 10-day volatility
            features.append(self._calculate_volatility(target_values, 30))  # 30-day volatility
            
            # Momentum features
            features.append(self._calculate_momentum(target_values, 5))   # 5-day momentum
            features.append(self._calculate_momentum(target_values, 20))  # 20-day momentum
            
            # Volume features (if available)
            if 'volume' in target_data.columns:
                volume_values = target_data['volume'].values
                features.append(volume_values)
                features.append(self._calculate_sma(volume_values, 10))  # Volume SMA
        else:
            features.append(target_values)
        
        # Combine all features
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = pd.DataFrame(feature_matrix).fillna(method='bfill').fillna(method='ffill').values
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.sequence_length, len(feature_matrix)):
            X.append(feature_matrix[i-self.sequence_length:i])
            y.append(target_values[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _calculate_sma(self, values, window):
        """Simple Moving Average"""
        return pd.Series(values).rolling(window=window).mean().values
    
    def _calculate_ema(self, values, span):
        """Exponential Moving Average"""
        return pd.Series(values).ewm(span=span).mean().values
    
    def _calculate_rsi(self, values, window=14):
        """Relative Strength Index"""
        delta = pd.Series(values).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def _calculate_macd(self, values, fast=12, slow=26):
        """MACD Indicator"""
        exp1 = pd.Series(values).ewm(span=fast).mean()
        exp2 = pd.Series(values).ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd.values
    
    def _calculate_bollinger_bands(self, values, window=20):
        """Bollinger Bands - returns % position within bands"""
        sma = pd.Series(values).rolling(window=window).mean()
        std = pd.Series(values).rolling(window=window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        # Calculate position within bands (0-1 scale)
        bb_position = (values - lower_band) / (upper_band - lower_band)
        return np.clip(bb_position.values, 0, 1)
    
    def _calculate_volatility(self, values, window):
        """Rolling volatility"""
        returns = pd.Series(values).pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility.values
    
    def _calculate_momentum(self, values, window):
        """Price momentum"""
        momentum = pd.Series(values).pct_change(window)
        return momentum.values
    
    def build_advanced_lstm_model(self, input_shape, model_type='lstm_attention'):
        """Build advanced LSTM architecture with attention mechanisms"""
        
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        if model_type == 'lstm_attention':
            # LSTM with Attention
            inputs = Input(shape=input_shape)
            
            # First LSTM layer
            lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            lstm1 = BatchNormalization()(lstm1)
            
            # Second LSTM layer
            lstm2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
            lstm2 = BatchNormalization()(lstm2)
            
            # Attention mechanism
            attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm2, lstm2)
            attention = Dropout(0.2)(attention)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
            
            # Dense layers
            dense1 = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(pooled)
            dense1 = Dropout(0.3)(dense1)
            
            dense2 = Dense(25, activation='relu', kernel_regularizer=l2(0.01))(dense1)
            dense2 = Dropout(0.2)(dense2)
            
            # Output layer
            outputs = Dense(1, activation='linear')(dense2)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        elif model_type == 'gru_ensemble':
            # GRU Ensemble Architecture
            inputs = Input(shape=input_shape)
            
            # Multiple GRU branches
            gru1 = GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(inputs)
            gru2 = GRU(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(inputs)
            
            # Combine branches
            combined = tf.keras.layers.concatenate([gru1, gru2])
            combined = BatchNormalization()(combined)
            
            # Dense layers
            dense1 = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(combined)
            dense1 = Dropout(0.3)(dense1)
            
            outputs = Dense(1, activation='linear')(dense1)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        else:  # Classic LSTM
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                Dense(50, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                Dense(25, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                Dense(1, activation='linear')
            ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_models(self, df, target_series, model_types=['lstm_attention', 'gru_ensemble', 'classic_lstm']):
        """Train multiple LSTM models and select the best one"""
        
        if not DEEP_LEARNING_AVAILABLE:
            st.error("🚫 TensorFlow required for LSTM models")
            return False
        
        st.info(f"🧠 Training advanced LSTM models for {target_series}...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(df, target_series)
        if X_train is None:
            st.warning(f"❌ Insufficient data for {target_series}")
            return False
        
        # Scale features
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Scale features
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Scale targets
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Store scalers
        self.feature_scalers[target_series] = feature_scaler
        self.scalers[target_series] = target_scaler
        
        # Train multiple model types
        best_model = None
        best_score = float('inf')
        model_results = {}
        
        for model_type in model_types:
            try:
                st.info(f"🔧 Training {model_type} model...")
                
                # Build model
                model = self.build_advanced_lstm_model(
                    input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                    model_type=model_type
                )
                
                if model is None:
                    continue
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, monitor='val_loss')
                ]
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train_scaled,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate model
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results[model_type] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'history': history.history
                }
                
                # Check if best model
                if mse < best_score:
                    best_score = mse
                    best_model = (model_type, model)
                
                st.success(f"✅ {model_type}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                st.warning(f"⚠️ Failed to train {model_type}: {str(e)}")
                continue
        
        if best_model:
            # Store best model
            self.models[target_series] = best_model[1]
            self.model_performance[target_series] = model_results
            
            st.success(f"🏆 Best model for {target_series}: {best_model[0]} (MSE: {best_score:.4f})")
            return True
        else:
            st.error(f"❌ No models trained successfully for {target_series}")
            return False
    
    def predict_lstm(self, df, target_series, periods=30):
        """Generate predictions using trained LSTM model"""
        
        if target_series not in self.models:
            st.warning(f"❌ No trained model for {target_series}")
            return None
        
        # Get recent data
        target_data = df[df['series_id'] == target_series].sort_values('date')
        recent_data = target_data.tail(self.sequence_length * 2)  # Extra data for features
        
        # Prepare features for the most recent sequence
        X_pred, _, _, _ = self.prepare_lstm_data(pd.concat([recent_data] * 2), target_series)
        if X_pred is None:
            return None
        
        # Use the last sequence
        last_sequence = X_pred[-1:]
        
        # Scale features
        feature_scaler = self.feature_scalers[target_series]
        target_scaler = self.scalers[target_series]
        
        last_sequence_reshaped = last_sequence.reshape(-1, last_sequence.shape[-1])
        last_sequence_scaled = feature_scaler.transform(last_sequence_reshaped).reshape(last_sequence.shape)
        
        # Generate predictions
        model = self.models[target_series]
        predictions_scaled = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(periods):
            # Predict next value
            pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            
            # Update sequence for next prediction
            # This is a simplified approach - in practice, you'd need to update all features
            new_features = current_sequence[0, -1:, :].copy()
            new_features[0, 0] = pred_scaled  # Update the price feature
            
            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1:, :] = new_features
        
        # Inverse transform predictions
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
        
        # Create prediction dates
        last_date = target_data.iloc[-1]['date']
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date).date()
        
        pred_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        
        return {
            'dates': pred_dates,
            'predictions': predictions.tolist(),
            'series_name': target_data.iloc[-1]['series_name'],
            'model_type': 'LSTM Neural Network',
            'confidence_interval': self._calculate_prediction_intervals(predictions, target_series)
        }
    
    def _calculate_prediction_intervals(self, predictions, target_series):
        """Calculate prediction confidence intervals"""
        
        if target_series in self.model_performance:
            # Use historical model performance to estimate uncertainty
            performance = self.model_performance[target_series]
            best_model_type = min(performance.keys(), key=lambda k: performance[k]['mae'])
            mae = performance[best_model_type]['mae']
            
            # Simple confidence interval based on MAE
            upper_bound = [p + 1.96 * mae for p in predictions]
            lower_bound = [p - 1.96 * mae for p in predictions]
            
            return {
                'upper': upper_bound,
                'lower': lower_bound,
                'mae_based': True
            }
        
        # Fallback to percentage-based intervals
        upper_bound = [p * 1.1 for p in predictions]
        lower_bound = [p * 0.9 for p in predictions]
        
        return {
            'upper': upper_bound,
            'lower': lower_bound,
            'mae_based': False
        }

class EnsembleLearningPredictor:
    """Ensemble learning combining LSTM with traditional ML models"""
    
    def __init__(self):
        self.lstm_predictor = AdvancedLSTMPredictor()
        self.traditional_models = {}
        self.ensemble_weights = {}
    
    def train_ensemble_models(self, df, target_series):
        """Train ensemble of LSTM and traditional ML models"""
        
        # Train LSTM models
        lstm_success = self.lstm_predictor.train_lstm_models(df, target_series)
        
        # Train traditional ML models
        traditional_success = self._train_traditional_models(df, target_series)
        
        if lstm_success or traditional_success:
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(df, target_series)
            return True
        
        return False
    
    def _train_traditional_models(self, df, target_series):
        """Train traditional ML models for ensemble"""
        
        target_data = df[df['series_id'] == target_series].sort_values('date')
        if len(target_data) < 50:
            return False
        
        # Prepare features for traditional ML
        values = target_data['value'].values
        X, y = [], []
        
        lookback = 30
        for i in range(lookback, len(values)):
            # Create feature vector
            features = []
            
            # Lagged values
            features.extend(values[i-lookback:i])
            
            # Technical indicators
            features.append(np.mean(values[i-5:i]))    # 5-day MA
            features.append(np.mean(values[i-20:i]))   # 20-day MA
            features.append(np.std(values[i-20:i]))    # 20-day volatility
            
            X.append(features)
            y.append(values[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', gamma='scale')
        }
        
        model_scores = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                model_scores[name] = {'model': model, 'mse': mse}
            except Exception as e:
                st.warning(f"⚠️ Failed to train {name}: {str(e)}")
        
        if model_scores:
            self.traditional_models[target_series] = model_scores
            return True
        
        return False
    
    def _calculate_ensemble_weights(self, df, target_series):
        """Calculate optimal ensemble weights based on model performance"""
        
        weights = {'lstm': 0.0, 'traditional': 0.0}
        
        # LSTM performance
        if target_series in self.lstm_predictor.model_performance:
            lstm_performance = self.lstm_predictor.model_performance[target_series]
            best_lstm_mae = min([perf['mae'] for perf in lstm_performance.values()])
            weights['lstm'] = 1.0 / (1.0 + best_lstm_mae)
        
        # Traditional ML performance
        if target_series in self.traditional_models:
            traditional_performance = self.traditional_models[target_series]
            best_traditional_mse = min([perf['mse'] for perf in traditional_performance.values()])
            weights['traditional'] = 1.0 / (1.0 + best_traditional_mse)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {'lstm': 0.5, 'traditional': 0.5}
        
        self.ensemble_weights[target_series] = weights
    
    def predict_ensemble(self, df, target_series, periods=30):
        """Generate ensemble predictions"""
        
        predictions_dict = {}
        
        # Get LSTM predictions
        if target_series in self.lstm_predictor.models:
            lstm_pred = self.lstm_predictor.predict_lstm(df, target_series, periods)
            if lstm_pred:
                predictions_dict['lstm'] = lstm_pred['predictions']
        
        # Get traditional ML predictions (simplified for now)
        if target_series in self.traditional_models:
            # This would implement traditional ML prediction
            # For now, use a simple trend-based prediction
            target_data = df[df['series_id'] == target_series].sort_values('date')
            recent_values = target_data['value'].tail(10).values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            traditional_pred = []
            last_value = recent_values[-1]
            for i in range(periods):
                traditional_pred.append(last_value + trend * (i + 1))
            
            predictions_dict['traditional'] = traditional_pred
        
        # Combine predictions using ensemble weights
        if predictions_dict and target_series in self.ensemble_weights:
            weights = self.ensemble_weights[target_series]
            
            ensemble_predictions = []
            for i in range(periods):
                weighted_sum = 0
                total_weight = 0
                
                for model_type, predictions in predictions_dict.items():
                    if i < len(predictions):
                        weight = weights.get(model_type, 0)
                        weighted_sum += predictions[i] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_predictions.append(weighted_sum / total_weight)
                else:
                    ensemble_predictions.append(predictions_dict['lstm'][i] if 'lstm' in predictions_dict else 0)
            
            # Create prediction result
            target_data = df[df['series_id'] == target_series].sort_values('date')
            last_date = target_data.iloc[-1]['date']
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date).date()
            
            pred_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            
            return {
                'dates': pred_dates,
                'predictions': ensemble_predictions,
                'series_name': target_data.iloc[-1]['series_name'],
                'model_type': 'Ensemble (LSTM + ML)',
                'individual_predictions': predictions_dict,
                'weights': weights
            }
        
        return None

# Model performance tracker
class ModelPerformanceTracker:
    """Track and compare model performance over time"""
    
    def __init__(self):
        self.performance_history = {}
    
    def log_prediction_accuracy(self, model_name, target_series, actual_values, predicted_values, prediction_date):
        """Log prediction accuracy for performance tracking"""
        
        if len(actual_values) != len(predicted_values):
            return
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        
        # Store performance
        key = f"{model_name}_{target_series}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append({
            'date': prediction_date,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'accuracy': 100 - mape
        })
    
    def get_model_rankings(self):
        """Get model performance rankings"""
        
        rankings = {}
        for key, history in self.performance_history.items():
            if history:
                # Calculate average performance
                avg_mae = np.mean([h['mae'] for h in history])
                avg_mape = np.mean([h['mape'] for h in history])
                avg_accuracy = np.mean([h['accuracy'] for h in history])
                
                rankings[key] = {
                    'avg_mae': avg_mae,
                    'avg_mape': avg_mape,
                    'avg_accuracy': avg_accuracy,
                    'num_predictions': len(history)
                }
        
        # Sort by accuracy
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)
        return sorted_rankings