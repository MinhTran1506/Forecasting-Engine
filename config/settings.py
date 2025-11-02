"""
Application configuration and settings
"""

import os

class Settings:
    # Application
    APP_NAME = "Future1 Pro"
    APP_VERSION = "2.0.0"
    
    # Security
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 5
    
    # Data
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    
    # Forecasting
    MIN_DATA_POINTS = 12
    MAX_FORECAST_PERIODS = 96
    DEFAULT_FORECAST_PERIODS = 12
    DEFAULT_CONFIDENCE_LEVEL = 95
    
    # Models
    AVAILABLE_MODELS = 17
    
    # Visualization
    CHART_HEIGHT = 500
    COMPARISON_CHART_HEIGHT = 600
    
    # Export
    EXCEL_ENGINE = 'xlsxwriter'
    
    # OAuth (Enterprise features)
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
    GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID', '')
    
    @classmethod
    def get_config(cls):
        return {
            'app_name': cls.APP_NAME,
            'version': cls.APP_VERSION,
            'models': cls.AVAILABLE_MODELS,
            'max_forecast': cls.MAX_FORECAST_PERIODS
        }