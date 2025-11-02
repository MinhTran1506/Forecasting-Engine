
import pandas as pd
import numpy as np
from utils.metrics import ForecastMetrics
from utils.visualizer import Visualizer
from models.all_models import ModelFactory
import io
import os
class DataProcessor:
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d'
        ]
    def convert_xlsx_to_csv(input_filepath, output_filepath=None, sheet_name=0):
        """
        Converts a single sheet from an Excel file (.xlsx) to a CSV file.

        Args:
            input_filepath (str): The path to the input Excel file.
            output_filepath (str, optional): The path for the output CSV file. 
                                            If None, it generates a path in the same directory.
            sheet_name (str or int, optional): The name or index of the Excel sheet to read.
                                                Defaults to 0 (the first sheet).

        Returns:
            str: The path to the generated CSV file, or None if conversion fails.
        """
        try:
            # 1. Read the Excel file into a pandas DataFrame
            df = pd.read_excel(input_filepath, sheet_name=sheet_name)
            
            # 2. Determine the output path if not provided
            if output_filepath is None:
                base, ext = os.path.splitext(input_filepath)
                output_filepath = base + '.csv'
                
            # 3. Write the DataFrame to a CSV file
            df.to_csv(output_filepath, index=False, encoding='utf-8')
            
            print(f"Successfully converted '{input_filepath}' (Sheet: {sheet_name}) to '{output_filepath}'")
            return output_filepath
            
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_filepath}'")
            return None
        except ValueError as e:
            print(f"Error reading Excel file (check sheet name/index): {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during conversion: {e}")
            return None
    
    def load_data(self, file_path_or_object):
        """Load data from a file path (string) or a file object."""
        try:
            # Check if input is a string (a file path)
            if isinstance(file_path_or_object, str):
                file_path = file_path_or_object
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            
            # Check if input is a file object (from Streamlit upload)
            elif hasattr(file_path_or_object, 'name'):
                file_obj = file_path_or_object
                if file_obj.name.lower().endswith('.csv'):
                    df = pd.read_csv(file_obj)
                else:
                    df = pd.read_excel(file_obj)
            
            else:
                # Handle unexpected input type
                print("Error: Input is neither a file path string nor a file object.")
                return None
                
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path_or_object}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}")
            return None
    def detect_date_column(self, df):
        """Auto-detect date column from dataframe"""
        date_keywords = ['date', 'billing', 'time', 'period', 'month', 'day', 'year']
        
        for col in df.columns:
            # Check by column name
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    pd.to_datetime(df[col])
                    return col
                except:
                    continue
            
            # Check by content
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
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
            # st.warning("Some dates could not be parsed. They will be excluded.")
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

processor = DataProcessor()
df = processor.load_data("D:\Coding\Forecasting-Engine\data\pivot.xlsx")
all_cols = df.columns.tolist()
date_col = processor.detect_date_column(df)
date_col_idx = all_cols.index(date_col) if date_col else 0
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(date_col_idx)
selected_date = all_cols[date_col_idx]
value_col = numeric_cols[-1]
agg_level = False
ts_data = processor.prepare_time_series(
                    df, selected_date, value_col, 
                    agg_level if agg_level else None
                )

y = ts_data[value_col].values
print(ts_data)