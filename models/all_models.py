"""
All forecasting models implementation
"""

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:
    """Factory class to create and train all forecasting models"""
    
    def __init__(self):
        self.models = {
            '1. Simple Average': self._simple_average,
            '2. Weighted Average': self._weighted_average,
            '3. Simple Moving Average': self._simple_moving_average,
            '4. Weighted Moving Average': self._weighted_moving_average,
            '5. Linear Regression': self._linear_regression,
            '6. Seasonal Linear Regression': self._seasonal_linear_regression,
            '7. Single Exponential Smoothing': self._single_exp_smoothing,
            '8. Double Exponential Smoothing': self._double_exp_smoothing,
            '9. Triple Exponential Smoothing': self._triple_exp_smoothing,
            '10. Automated Exp Smoothing': self._automated_exp_smoothing,
            '11. Adaptive Response Rate': self._adaptive_response_rate,
            '12. Browns Linear Exp Smoothing': self._browns_linear,
            '13. Auto-ARIMA': self._auto_arima,
            '14. SARIMAX': self._sarimax,
            '15. Gradient Boosting': self._gradient_boosting,
            '16. XGBoost-like (GB variant)': self._xgboost_variant,
            '17. Prophet': self._prophet
        }
    
    def get_all_model_names(self):
        return list(self.models.keys())
    
    def train_and_predict(self, model_name, train_data, periods):
        """Train model and generate predictions"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name](train_data, periods)
    
    def _simple_average(self, train, periods):
        """Simple Average method"""
        avg = np.mean(train)
        predictions = np.full(periods, avg)
        return {'predictions': predictions, 'model': None}
    
    def _weighted_average(self, train, periods):
        """Weighted Average (more weight to recent data)"""
        weights = np.arange(1, len(train) + 1)
        avg = np.average(train, weights=weights)
        predictions = np.full(periods, avg)
        return {'predictions': predictions, 'model': None}
    
    def _simple_moving_average(self, train, periods):
        """Simple Moving Average"""
        window = min(3, len(train))
        avg = np.mean(train[-window:])
        predictions = np.full(periods, avg)
        return {'predictions': predictions, 'model': None}
    
    def _weighted_moving_average(self, train, periods):
        """Weighted Moving Average"""
        window = min(3, len(train))
        weights = np.arange(1, window + 1)
        avg = np.average(train[-window:], weights=weights)
        predictions = np.full(periods, avg)
        return {'predictions': predictions, 'model': None}
    
    def _linear_regression(self, train, periods):
        """Linear Regression"""
        X = np.arange(len(train)).reshape(-1, 1)
        y = train
        model = LinearRegression()
        model.fit(X, y)
        
        X_future = np.arange(len(train), len(train) + periods).reshape(-1, 1)
        predictions = model.predict(X_future)
        return {'predictions': predictions, 'model': model}
    
    def _seasonal_linear_regression(self, train, periods):
        """Seasonal Linear Regression"""
        X = np.arange(len(train)).reshape(-1, 1)
        # Add seasonal component (month of year)
        seasons = np.array([i % 12 for i in range(len(train))]).reshape(-1, 1)
        X_combined = np.hstack([X, seasons])
        
        model = LinearRegression()
        model.fit(X_combined, train)
        
        X_future = np.arange(len(train), len(train) + periods).reshape(-1, 1)
        seasons_future = np.array([i % 12 for i in range(len(train), len(train) + periods)]).reshape(-1, 1)
        X_future_combined = np.hstack([X_future, seasons_future])
        
        predictions = model.predict(X_future_combined)
        return {'predictions': predictions, 'model': model}
    
    def _single_exp_smoothing(self, train, periods):
        """Single Exponential Smoothing"""
        model = SimpleExpSmoothing(train).fit()
        predictions = model.forecast(periods)
        return {'predictions': predictions, 'model': model}
    
    def _double_exp_smoothing(self, train, periods):
        """Double Exponential Smoothing (Holt's method)"""
        model = Holt(train).fit()
        predictions = model.forecast(periods)
        return {'predictions': predictions, 'model': model}
    
    def _triple_exp_smoothing(self, train, periods):
        """Triple Exponential Smoothing (Holt-Winters)"""
        if len(train) >= 24:
            model = ExponentialSmoothing(
                train, seasonal='add', seasonal_periods=12, 
                trend='add'
            ).fit()
        else:
            model = ExponentialSmoothing(
                train, seasonal='add', seasonal_periods=3, 
                trend='add'
            ).fit()
        predictions = model.forecast(periods)
        return {'predictions': predictions, 'model': model}
    
    def _automated_exp_smoothing(self, train, periods):
        """Automated Exponential Smoothing (optimized parameters)"""
        if len(train) >= 24:
            model = ExponentialSmoothing(
                train, seasonal='add', seasonal_periods=12, 
                trend='add', initialization_method="estimated"
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                train, seasonal='add', seasonal_periods=3, 
                trend='add', initialization_method="estimated"
            ).fit(optimized=True)
        predictions = model.forecast(periods)
        return {'predictions': predictions, 'model': model}
    
    def _adaptive_response_rate(self, train, periods):
        """Adaptive Response Rate Single Exponential Smoothing"""
        # Adaptive alpha based on forecast error
        alpha = 0.3
        forecast = [train[0]]
        
        for i in range(1, len(train)):
            error = abs(train[i] - forecast[-1])
            # Adapt alpha based on error magnitude
            adaptive_alpha = min(0.9, alpha + 0.1 * (error / np.mean(train[:i])))
            new_forecast = adaptive_alpha * train[i] + (1 - adaptive_alpha) * forecast[-1]
            forecast.append(new_forecast)
        
        # Generate future predictions
        last_forecast = forecast[-1]
        predictions = np.full(periods, last_forecast)
        return {'predictions': predictions, 'model': None}
    
    def _browns_linear(self, train, periods):
        """Brown's Linear Exponential Smoothing"""
        alpha = 0.3
        s1 = [train[0]]
        s2 = [train[0]]
        
        for i in range(1, len(train)):
            s1_new = alpha * train[i] + (1 - alpha) * s1[-1]
            s2_new = alpha * s1_new + (1 - alpha) * s2[-1]
            s1.append(s1_new)
            s2.append(s2_new)
        
        # Calculate trend
        a = 2 * s1[-1] - s2[-1]
        b = (alpha / (1 - alpha)) * (s1[-1] - s2[-1])
        
        # Generate predictions
        predictions = np.array([a + b * (h + 1) for h in range(periods)])
        return {'predictions': predictions, 'model': None}
    
    def _auto_arima(self, train, periods):
        """Auto-ARIMA (simplified version with fixed parameters)"""
        # Try different ARIMA orders and select best
        best_aic = np.inf
        best_model = None
        
        # Test common configurations
        orders = [(0,0,0), (1,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
        
        for order in orders:
            try:
                model = ARIMA(train, order=order).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
            except:
                continue
        
        if best_model is None:
            # Fallback to simple model
            best_model = ARIMA(train, order=(1,0,0)).fit()
        
        predictions = best_model.forecast(steps=periods)
        return {'predictions': predictions, 'model': best_model}
    
    def _sarimax(self, train, periods):
        """SARIMAX model"""
        try:
            if len(train) >= 24:
                # Seasonal ARIMA
                model = ARIMA(train, order=(1,0,1), seasonal_order=(1,0,1,12)).fit()
            else:
                # Non-seasonal ARIMA
                model = ARIMA(train, order=(1,0,1)).fit()
            
            predictions = model.forecast(steps=periods)
            return {'predictions': predictions, 'model': model}
        except:
            # Fallback to simpler model
            return self._auto_arima(train, periods)
    
    def _gradient_boosting(self, train, periods):
        """Gradient Boosting Regressor"""
        # Create features (lagged values + trend)
        n_lags = min(3, len(train) - 1)
        
        X = []
        y = []
        for i in range(n_lags, len(train)):
            X.append(list(train[i-n_lags:i]) + [i])
            y.append(train[i])
        
        X = np.array(X)
        y = np.array(y)
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)
        
        # Generate predictions
        predictions = []
        last_values = list(train[-n_lags:])
        
        for i in range(periods):
            features = last_values[-n_lags:] + [len(train) + i]
            pred = model.predict([features])[0]
            predictions.append(pred)
            last_values.append(pred)
        
        return {'predictions': np.array(predictions), 'model': model}
    
    def _xgboost_variant(self, train, periods):
        """XGBoost-like variant using Gradient Boosting with different params"""
        n_lags = min(5, len(train) - 1)
        
        X = []
        y = []
        for i in range(n_lags, len(train)):
            X.append(list(train[i-n_lags:i]) + [i, i % 12])  # Add seasonal component
            y.append(train[i])
        
        X = np.array(X)
        y = np.array(y)
        
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        model.fit(X, y)
        
        # Generate predictions
        predictions = []
        last_values = list(train[-n_lags:])
        
        for i in range(periods):
            time_idx = len(train) + i
            features = last_values[-n_lags:] + [time_idx, time_idx % 12]
            pred = model.predict([features])[0]
            predictions.append(pred)
            last_values.append(pred)
        
        return {'predictions': np.array(predictions), 'model': model}
    
    def _prophet(self, train, periods):
        """Facebook Prophet"""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(train), freq='MS'),
            'y': train
        })
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(df)
        
        # Generate future dataframe
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        predictions = forecast['yhat'].values[-periods:]
        return {'predictions': predictions, 'model': model}