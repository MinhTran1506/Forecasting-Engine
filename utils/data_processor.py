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

    def remove_outliers(self, series, threshold=3.0):
        """Remove outliers using z-score method"""
        mean = np.mean(series)
        std = np.std(series)
        z_scores = np.abs((series - mean) / std)
        mask = z_scores < threshold
        
        removed = len(series) - mask.sum()
        if removed > 0:
            st.info(f"📊 Removed {removed} outliers ({removed/len(series)*100:.1f}% of data)")
        
        return series[mask], mask
    
    def smooth_series(self, series, window=3):
        """Apply moving average smoothing"""
        smoothed = pd.Series(series).rolling(window=window, center=True).mean()
        # Fill NaN values at edges
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
        return smoothed.values
    
    def log_transform_series(self, series):
        """Apply log transformation (handles zeros)"""
        # Add small constant to avoid log(0)
        min_val = series.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            series = series + shift
        return np.log(series)
    
    def inverse_log_transform(self, series, shift=0):
        """Inverse log transformation"""
        return np.exp(series) - shift
    
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
        # First, try pandas auto-detection (most robust)
        try:
            parsed = pd.to_datetime(date_series, infer_datetime_format=True)
            if parsed.notna().sum() / len(parsed) >= 0.8:  # 80% success rate
                return parsed
        except:
            pass
        
        # Try common formats explicitly
        for fmt in self.date_formats:
            try:
                parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
                if parsed.notna().sum() / len(parsed) >= 0.8:
                    return parsed
            except:
                continue

        try:
            return pd.to_datetime(date_series, errors='coerce')
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
        
        # Parse dates with improved handling
        st.info(f"📅 Parsing dates from column: {date_col}")
        df_copy[date_col] = self.parse_date(df_copy[date_col])
        
        # Check for parsing failures
        failed_dates = df_copy[date_col].isna().sum()
        if failed_dates > 0:
            st.warning(f"⚠️ Could not parse {failed_dates} dates ({failed_dates/len(df_copy)*100:.1f}%). These rows will be excluded.")
            df_copy = df_copy.dropna(subset=[date_col])
        
        if len(df_copy) == 0:
            st.error("❌ No valid dates found. Please check your date column format.")
            return None
        
        st.success(f"✅ Successfully parsed {len(df_copy)} dates")
        
        # Show date range
        min_date = df_copy[date_col].min()
        max_date = df_copy[date_col].max()
        st.info(f"📊 Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Aggregation
        if agg_cols:
            st.info(f"🔄 Aggregating by: {', '.join([date_col] + agg_cols)}")
            group_cols = [date_col] + agg_cols
            ts_data = df_copy.groupby(group_cols)[value_col].sum().reset_index()
        else:
            st.info(f"🔄 Aggregating by: {date_col}")
            ts_data = df_copy.groupby(date_col)[value_col].sum().reset_index()
        
        # Sort by date
        ts_data = ts_data.sort_values(date_col).reset_index(drop=True)
        
        st.success(f"✅ Aggregated to {len(ts_data)} time periods")
        
        # Display sample of aggregated data
        with st.expander("👀 Preview Aggregated Data"):
            st.dataframe(ts_data.head(10), use_container_width=True)
        
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
