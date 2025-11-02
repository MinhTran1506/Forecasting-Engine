"""
Base forecasting model class
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseForecaster(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.fitted_values = None
        self.forecast_values = None
    
    @abstractmethod
    def fit(self, data):
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, periods):
        """Generate forecast for specified periods"""
        pass
    
    def get_fitted_values(self):
        """Return fitted values"""
        return self.fitted_values
    
    def get_forecast(self):
        """Return forecast values"""
        return self.forecast_values