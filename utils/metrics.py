"""
Forecast accuracy metrics
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ForecastMetrics:
    @staticmethod
    def mape(actual, predicted):
        """Mean Absolute Percentage Error"""
        actual, predicted = np.array(actual), np.array(predicted)
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    @staticmethod
    def mae(actual, predicted):
        """Mean Absolute Error"""
        return mean_absolute_error(actual, predicted)
    
    @staticmethod
    def rmse(actual, predicted):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(actual, predicted))
    
    @staticmethod
    def mse(actual, predicted):
        """Mean Squared Error"""
        return mean_squared_error(actual, predicted)
    
    @staticmethod
    def r2(actual, predicted):
        """R-squared Score"""
        return r2_score(actual, predicted)
    
    @staticmethod
    def smape(actual, predicted):
        """Symmetric Mean Absolute Percentage Error"""
        actual, predicted = np.array(actual), np.array(predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0
        return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def calculate_all(actual, predicted):
        """Calculate all metrics"""
        return {
            'MAPE': round(ForecastMetrics.mape(actual, predicted), 2),
            'MAE': round(ForecastMetrics.mae(actual, predicted), 2),
            'RMSE': round(ForecastMetrics.rmse(actual, predicted), 2),
            'MSE': round(ForecastMetrics.mse(actual, predicted), 2),
            'R²': round(ForecastMetrics.r2(actual, predicted), 4),
            'SMAPE': round(ForecastMetrics.smape(actual, predicted), 2)
        }