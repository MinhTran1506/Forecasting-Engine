"""
Data processing and preparation utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d'
        ]
    
    def load_data(self, file):
        """Load data from CSV or Excel file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def load_default_data(self):
        """Load default example data"""
        try:
            return pd.read_csv("Holts-Winter-data-input-VNHOLSC064.csv", index_col=False)
        except:
            # Create sample data if file doesn't exist
            dates = pd.date_range('2020-01-01', periods=36, freq='MS')
            data = {
                'Date': dates,
                'Vol': np.random.randint(100, 1000, 36)
            }
            return pd.DataFrame(data)
    
    def detect_date_column(self, df):
        """Auto-detect date column from dataframe - Content first, then name"""
        date_keywords = ['date', 'billing', 'time', 'period', 'month', 'day', 'year']
        
        # First pass: Check all columns by actual content (most reliable)
        for col in df.columns:
            # Skip numeric columns (they're unlikely to be dates)
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            try:
                # Try to parse the column as datetime
                parsed = pd.to_datetime(df[col], errors='coerce')
                # Check if at least 80% of non-null values were successfully parsed
                valid_dates = parsed.notna().sum()
                total_non_null = df[col].notna().sum()
                
                if total_non_null > 0 and (valid_dates / total_non_null) >= 0.8:
                    return col
            except:
                continue
        
        # Second pass: Check by column name (as fallback)
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    # Verify the content is actually parseable as dates
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    valid_dates = parsed.notna().sum()
                    total_non_null = df[col].notna().sum()
                    
                    if total_non_null > 0 and (valid_dates / total_non_null) >= 0.8:
                        return col
                except:
                    continue
        
        return None
    
    def parse_date(self, date_series):
        """Parse date series with multiple format attempts"""
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_series, format=fmt)
            except:
                continue
        
        # If all fail, use pandas auto-detection
        try:
            return pd.to_datetime(date_series)
        except:
            return None
    
    def prepare_time_series(self, df, date_col, value_col, agg_cols=None):
        """
        Prepare and aggregate time series data
        
        Args:
            df: DataFrame
            date_col: Name of date column
            value_col: Name of value column to forecast
            agg_cols: List of columns to group by (optional)
        
        Returns:
            Aggregated time series DataFrame
        """
        df_copy = df.copy()
        
        # Parse dates
        df_copy[date_col] = self.parse_date(df_copy[date_col])
        
        if df_copy[date_col].isna().any():
            st.warning("Some dates could not be parsed. They will be excluded.")
            df_copy = df_copy.dropna(subset=[date_col])
        
        # Aggregation
        if agg_cols:
            group_cols = [date_col] + agg_cols
            ts_data = df_copy.groupby(group_cols)[value_col].sum().reset_index()
        else:
            ts_data = df_copy.groupby(date_col)[value_col].sum().reset_index()
        
        # Sort by date
        ts_data = ts_data.sort_values(date_col).reset_index(drop=True)
        
        return ts_data
    
    def validate_time_series(self, df, date_col, value_col):
        """Validate if data is suitable for time series forecasting"""
        issues = []
        
        # Check minimum length
        if len(df) < 12:
            issues.append(f"⚠️ Only {len(df)} observations. Recommend at least 12 for reliable forecasts.")
        
        # Check for missing values
        missing = df[value_col].isna().sum()
        if missing > 0:
            issues.append(f"⚠️ {missing} missing values detected in target column.")
        
        # Check for negative values
        if (df[value_col] < 0).any():
            issues.append("⚠️ Negative values detected. Some models may not work properly.")
        
        # Check for zeros
        zeros = (df[value_col] == 0).sum()
        if zeros > len(df) * 0.1:
            issues.append(f"⚠️ {zeros} zero values ({zeros/len(df)*100:.1f}%). May affect model performance.")
        
        return issues
