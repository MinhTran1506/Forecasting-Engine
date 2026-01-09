"""
Multi-model comparison page with all 17+ models
Enhanced with:
- Period-based time series (Year + Week/Month)
- Group-wise model training (each model for each group)
- Negative value filtering
- Input boxes for forecast settings
- Smart train/test split
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.metrics import ForecastMetrics
from utils.visualizer import Visualizer
from models.all_models import ModelFactory
import io

def create_period_column(df, year_col, time_col):
    """
    Create sequential period column from Year + Week/Month
    Example: Year=[2024,2024,2025,2025], Week=[1,2,1,2] -> Period=[1,2,3,4]
    """
    df_sorted = df.sort_values([year_col, time_col]).copy()
    
    # Create unique year-time combinations
    df_sorted['_temp_year_time'] = df_sorted[year_col].astype(str) + '_' + df_sorted[time_col].astype(str)
    
    # Get unique periods in order
    unique_periods = df_sorted['_temp_year_time'].unique()
    period_map = {period: idx + 1 for idx, period in enumerate(unique_periods)}
    
    # Map to sequential periods
    df_sorted['Period'] = df_sorted['_temp_year_time'].map(period_map)
    df_sorted = df_sorted.drop('_temp_year_time', axis=1)
    
    return df_sorted

def aggregate_forecasts_from_groups(matching_groups, group_results, settings, train_ratio, min_periods):
    """
    Aggregate forecasts from individual group best-fit models instead of training a new model.
    Uses stored future forecasts from training phase for consistency.
    Returns: aggregated_y, aggregated_future_forecast, best_model_info
    """
    # Collect historical data by period from all matching groups
    period_data = {}
    for group_name in matching_groups:
        result = group_results[group_name]
        for period_idx, value in enumerate(result['full_data']):
            if period_idx not in period_data:
                period_data[period_idx] = []
            period_data[period_idx].append(value)
    
    if not period_data:
        return None, None, None
    
    # Sum values for each period
    aggregated_y = np.array([np.sum(period_data[p]) for p in sorted(period_data.keys())])
    
    # Aggregate forecasts from each group's best-fit model (use stored forecasts)
    forecast_by_period = {}
    model_names_used = {}
    group_forecasts = {}  # Store individual group forecasts for debugging
    
    for group_name in matching_groups:
        result = group_results[group_name]
        model_name = result['best_model_name']
        
        # Use the stored future forecast from training phase
        future_forecast = result.get('future_forecast', np.array([]))
        
        if len(future_forecast) > 0:
            group_forecasts[group_name] = future_forecast  # Store for reference
            
            # Track which models are being used
            if model_name not in model_names_used:
                model_names_used[model_name] = 0
            model_names_used[model_name] += 1
            
            # Add to forecast aggregation
            for period_idx, value in enumerate(future_forecast):
                if period_idx not in forecast_by_period:
                    forecast_by_period[period_idx] = []
                forecast_by_period[period_idx].append(value)
    
    # Sum forecasts across groups
    aggregated_future_forecast = np.array([
        np.sum(forecast_by_period[p]) for p in sorted(forecast_by_period.keys())
    ]) if forecast_by_period else np.array([])
    
    # Get best model info: model with highest frequency among groups
    if model_names_used:
        best_model = max(model_names_used.items(), key=lambda x: x[1])[0]
    else:
        best_model = "Aggregated Ensemble"
    
    return aggregated_y, aggregated_future_forecast, {
        'model_name': best_model,
        'models_used': model_names_used,
        'num_groups': len(matching_groups),
        'group_forecasts': group_forecasts
    }

def render(df):
    st.header("Best-Fit Forecast")
    st.markdown("**Train 10+ models for each planning level and let the system recommend the one with best performance**")
    
    if df is None:
        st.warning("⚠️ Please upload data to use Best-Fit Forecast")
        return
    
    processor = DataProcessor()
    
    # ========== STEP 1: DATA CONFIGURATION ==========
    st.subheader("⚙️ Data Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    all_cols = df.columns.tolist()
    
    with col1:
        # Detect Year column
        year_candidates = [c for c in all_cols if 'year' in c.lower()]
        year_col = st.selectbox(
            "📅 Year Column",
            all_cols,
            index=all_cols.index(year_candidates[0]) if year_candidates else 0,
            help="Column containing year values"
        )
    
    with col2:
        # Detect Week/Month column
        time_candidates = [c for c in all_cols if any(x in c.lower() for x in ['week', 'month', 'period'])]
        time_col = st.selectbox(
            "🔢 Week/Month Column",
            all_cols,
            index=all_cols.index(time_candidates[0]) if time_candidates else 0,
            help="Column containing week or month values"
        )
    
    with col3:
        # Select value column (quantity)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Prioritize columns with 'qty', 'quantity', 'vol', 'volume' in name
        qty_candidates = [c for c in numeric_cols if any(x in c.lower() for x in ['qty', 'quantity', 'vol', 'volume', 'sum'])]
        default_value_idx = numeric_cols.index(qty_candidates[0]) if qty_candidates else 0
        
        value_col = st.selectbox(
            "📊 Value Column (to forecast)",
            numeric_cols,
            index=default_value_idx,
            help="Numeric column containing quantities to forecast"
        )
    
    # ========== STEP 2: GROUPING CONFIGURATION ==========
    st.markdown("### 🎚️ Forecast Granularity")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Exclude year and time columns from categorical
    categorical_cols = [c for c in categorical_cols if c not in [year_col, time_col]]
    
    if categorical_cols:
        group_cols = st.multiselect(
            "📦 Group By (e.g., DC, Region, Product)",
            categorical_cols,
            help="Train separate models for each combination. Leave empty for overall forecast."
        )
    else:
        group_cols = []
        st.info("No categorical columns detected. Will forecast overall data.")
    
    # ========== STEP 3: FORECAST SETTINGS (INPUT BOXES) ==========
    st.markdown("### ⚙️ Forecast Settings")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        forecast_periods = st.number_input(
            "🔮 Forecast Periods",
            min_value=1,
            max_value=120,
            value=24,
            step=1,
            help="Number of periods to forecast ahead"
        )
    
    with col2:
        train_ratio = st.number_input(
            "📊 Train Ratio (%)",
            min_value=50,
            max_value=90,
            value=80,
            step=5,
            help="Percentage of data for training (rest for testing)"
        )
    
    with col3:
        confidence_level = st.number_input(
            "📈 Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1,
            help="Confidence interval for predictions"
        )
    
    with col4:
        min_periods = st.number_input(
            "⏰ Min Periods Required",
            min_value=3,
            max_value=240,
            value=12,
            step=1,
            help="Minimum data points required per group"
        )
    
    # ========== STEP 4: PREPROCESSING OPTIONS ==========
    with st.expander("🔧 Advanced Preprocessing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_outliers = st.checkbox(
                "Outliers correction",
                value=True,
                help="Correct extreme values that may distort forecasts"
            )
            
            if remove_outliers:
                outlier_std = st.slider(
                    "Outlier Threshold (Std Dev)",
                    1.5, 4.0, 2.0, 0.5,
                    help="Values beyond this standard deviation will be pulled back to the upper/lower limit value"
                )
        
        with col2:
            smooth_data = st.checkbox(
                "Smooth Data",
                value=False,
                help="Apply moving average smoothing"
            )
            
            if smooth_data:
                smooth_window = st.slider(
                    "Smoothing Window",
                    3, 12, 3,
                    help="Larger windows = more smoothing"
                )
    
    # ========== STEP 5: MODEL SELECTION ==========
    st.markdown("### 🛠️ Models Factory")
    model_selection = st.radio(
        "Choose models to train:",
        ["All Models (17)", "Fast Models Only (10)", "Linh's Favorites ✨ (5)", "Custom Selection"],
        horizontal=True
    )
    
    if model_selection == "Custom Selection":
        available_models = [
            "SARIMAX", "Gradient Boosting", "XGBoost", "Prophet", "Automated Exp Smoothing",
            "Auto-ARIMA", "Simple Average", "Weighted Average", "Simple Moving Average",
            "Weighted Moving Average", "Linear Regression", "Seasonal Linear Regression",
            "Single Exp Smoothing", "Double Exp Smoothing", "Triple Exp Smoothing",
            "Adaptive Response Rate", "Browns Linear"
        ]
        selected_models = st.multiselect(
            "Select models:",
            available_models,
            default=available_models[:5]
        )
    
    # ========== STEP 6: RUN FORECASTING ==========
    if st.button("🚀 Train Models & Generate Forecasts", type="primary", use_container_width=True):
        with st.spinner('🔄 Processing data and training models...'):
            try:
                # Create Period column
                df_processed = create_period_column(df.copy(), year_col, time_col)
                
                # Filter negative values
                original_len = len(df_processed)
                df_processed = df_processed[df_processed[value_col] >= 0]
                negative_filtered = original_len - len(df_processed)
                
                if negative_filtered > 0:
                    st.info(f"✨ Filtered {negative_filtered} negative values ({negative_filtered/original_len*100:.1f}%)")
                
                # Aggregate by Period and groups
                if group_cols:
                    agg_df = df_processed.groupby(['Period'] + group_cols)[value_col].sum().reset_index()
                else:
                    agg_df = df_processed.groupby('Period')[value_col].sum().reset_index()
                
                # Sort by Period
                agg_df = agg_df.sort_values('Period').reset_index(drop=True)
                
                st.success(f"✅ Data prepared: {len(agg_df)} time series observations")
                
                # Determine model names
                factory = ModelFactory()
                if model_selection == "All Models (17)":
                    model_names = factory.get_all_model_names()
                elif model_selection == "Fast Models Only (10)":
                    model_names = [m for m in factory.get_all_model_names()
                                 if not any(x in m for x in ['ARIMA', 'SARIMAX', 'Prophet', 'Gradient', 'XGBoost'])]
                elif model_selection == "Linh's Favorites ✨ (5)":
                    model_names = [
                        '10. Automated Exp Smoothing',
                        '14. SARIMAX',
                        '15. Gradient Boosting',
                        '16. XGBoost-like (GB variant)',
                        '17. Prophet'
                    ]
                else:  # Custom
                    name_map = {
                        'Simple Average': '1. Simple Average',
                        'Weighted Average': '2. Weighted Average',
                        'Simple Moving Average': '3. Simple Moving Average',
                        'Weighted Moving Average': '4. Weighted Moving Average',
                        'Linear Regression': '5. Linear Regression',
                        'Seasonal Linear Regression': '6. Seasonal Linear Regression',
                        'Single Exp Smoothing': '7. Single Exponential Smoothing',
                        'Double Exp Smoothing': '8. Double Exponential Smoothing',
                        'Triple Exp Smoothing': '9. Triple Exponential Smoothing',
                        'Automated Exp Smoothing': '10. Automated Exp Smoothing',
                        'Adaptive Response Rate': '11. Adaptive Response Rate',
                        'Browns Linear': '12. Browns Linear Exp Smoothing',
                        'Auto-ARIMA': '13. Auto-ARIMA',
                        'SARIMAX': '14. SARIMAX',
                        'Gradient Boosting': '15. Gradient Boosting',
                        'XGBoost': '16. XGBoost-like (GB variant)',
                        'Prophet': '17. Prophet'
                    }
                    model_names = [name_map[m] for m in selected_models if m in name_map]
                
                # ========== GROUP-WISE TRAINING ==========
                if group_cols:
                    # Get unique groups
                    group_combinations = agg_df[group_cols].drop_duplicates().reset_index(drop=True)
                    st.info(f"🔄 Training {len(model_names)} models for {len(group_combinations)} groups = {len(model_names) * len(group_combinations)} total models")
                    
                    all_group_results = {}
                    skipped_groups = []  # Track skipped groups
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_iterations = len(group_combinations)
                    
                    for group_idx, group_row in group_combinations.iterrows():
                        # Filter data for this group
                        group_filter = pd.Series([True] * len(agg_df))
                        group_name_parts = []
                        
                        for col in group_cols:
                            group_filter = group_filter & (agg_df[col] == group_row[col])
                            group_name_parts.append(f"{col}={group_row[col]}")
                        
                        group_name = ", ".join(group_name_parts)
                        group_data = agg_df[group_filter]
                        
                        # Check minimum periods
                        if len(group_data) < min_periods:
                            skipped_groups.append({
                                'group': group_name,
                                'periods': len(group_data),
                                'required': min_periods
                            })
                            continue
                        
                        status_text.text(f"Training models for: {group_name} ({group_idx + 1}/{total_iterations})")
                        
                        y = group_data[value_col].values
                        
                        # Apply preprocessing
                        if remove_outliers:
                            y, mask = processor.remove_outliers(y, outlier_std)
                        
                        if smooth_data:
                            y = processor.smooth_series(y, smooth_window)
                        
                        # Train/test split
                        test_size = int(len(y) * (1 - train_ratio / 100))
                        test_size = max(1, min(test_size, len(y) - min_periods))  # Ensure valid split
                        
                        train_data = y[:-test_size]
                        test_data = y[-test_size:]
                        
                        # Train all models for this group
                        group_results = {}
                        for model_name in model_names:
                            try:
                                result = factory.train_and_predict(model_name, train_data, len(test_data))
                                
                                if result is not None:
                                    # Ensure no negative predictions
                                    predictions = np.maximum(result['predictions'], 0)
                                    
                                    metrics = ForecastMetrics.calculate_all(test_data, predictions)
                                    group_results[model_name] = {
                                        'predictions': predictions,
                                        'metrics': metrics,
                                        'model': result.get('model')
                                    }
                            except Exception as e:
                                pass  # Skip failed models
                        
                        if group_results:
                            # Find best model for this group
                            best_model = min(group_results.items(), key=lambda x: x[1]['metrics']['MAPE'])
                            best_model_name = best_model[0]
                            
                            # Generate future forecast for best model using full data
                            future_result = factory.train_and_predict(
                                best_model_name,
                                y,
                                forecast_periods
                            )
                            future_forecast = np.maximum(future_result['predictions'], 0) if future_result else np.array([])
                            
                            all_group_results[group_name] = {
                                'best_model_name': best_model_name,
                                'best_model_result': best_model[1],
                                'all_results': group_results,
                                'train_data': train_data,
                                'test_data': test_data,
                                'full_data': y,
                                'group_filter': group_row.to_dict(),
                                'future_forecast': future_forecast  # Store for aggregation
                            }
                        
                        progress_bar.progress((group_idx + 1) / total_iterations)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display skipped groups in an expander if any
                    if skipped_groups:
                        with st.expander(f"⚠️ Skipped Groups ({len(skipped_groups)})"):
                            st.warning(f"The following {len(skipped_groups)} group(s) were skipped due to insufficient data:")
                            skipped_df = pd.DataFrame(skipped_groups)
                            skipped_df.columns = ['Group', 'Available Periods', 'Required Periods']
                            st.dataframe(skipped_df, use_container_width=True, hide_index=True)
                    
                    # Store results
                    st.session_state.group_results = all_group_results
                    st.session_state.forecast_settings = {
                        'forecast_periods': forecast_periods,
                        'group_cols': group_cols,
                        'value_col': value_col,
                        'factory': factory,
                        'agg_df': agg_df
                    }
                    
                else:
                    # Overall forecast (no grouping)
                    st.info(f"🔄 Training {len(model_names)} models for overall data")
                    
                    y = agg_df[value_col].values
                    
                    # Apply preprocessing
                    if remove_outliers:
                        y, mask = processor.remove_outliers(y, outlier_std)
                    
                    if smooth_data:
                        y = processor.smooth_series(y, smooth_window)
                    
                    # Train/test split
                    test_size = int(len(y) * (1 - train_ratio / 100))
                    test_size = max(1, min(test_size, len(y) - min_periods))
                    
                    train_data = y[:-test_size]
                    test_data = y[-test_size:]
                    
                    # Train models
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    all_results = {}
                    
                    for idx, model_name in enumerate(model_names):
                        status_text.text(f"Training {model_name}... ({idx+1}/{len(model_names)})")
                        
                        try:
                            result = factory.train_and_predict(model_name, train_data, len(test_data))
                            
                            if result is not None:
                                predictions = np.maximum(result['predictions'], 0)
                                metrics = ForecastMetrics.calculate_all(test_data, predictions)
                                all_results[model_name] = {
                                    'predictions': predictions,
                                    'metrics': metrics,
                                    'model': result.get('model')
                                }
                        except Exception as e:
                            pass
                        
                        progress_bar.progress((idx + 1) / len(model_names))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['metrics']['MAPE'])
                    
                    st.session_state.overall_results = {
                        'all_results': all_results,
                        'sorted_models': sorted_models,
                        'train_data': train_data,
                        'test_data': test_data,
                        'full_data': y
                    }
                    st.session_state.forecast_settings = {
                        'forecast_periods': forecast_periods,
                        'value_col': value_col,
                        'factory': factory,
                        'agg_df': agg_df
                    }
                
            except Exception as e:
                st.error(f"Error in forecasting: {e}")
                st.exception(e)
    
    # ========== DISPLAY RESULTS ==========
    viz = Visualizer()
    
    # Group-wise results
    if 'group_results' in st.session_state:
        st.markdown("---")
        st.subheader("🏆 Group-wise Model Performance")
        
        group_results = st.session_state.group_results
        settings = st.session_state.forecast_settings
        
        # Summary table
        summary_data = []
        for group_name, result in group_results.items():
            summary_data.append({
                'Group': group_name,
                'Best Model': result['best_model_name'],
                'MAPE (%)': result['best_model_result']['metrics']['MAPE'],
                'MAE': result['best_model_result']['metrics']['MAE'],
                'RMSE': result['best_model_result']['metrics']['RMSE'],
                'R²': result['best_model_result']['metrics']['R²']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # ========== INTERACTIVE FORECAST EXPLORER ==========
        st.markdown("---")
        st.markdown("## 🔍 Interactive Forecast Explorer")
        st.markdown("**Filter the visualization below to explore different scenarios**")
        
        # Get the group columns that were actually used
        active_group_cols = settings.get('group_cols', [])
        
        if active_group_cols:
            # Extract unique values for each group column from group_results
            available_filters = {}
            for group_name in group_results.keys():
                # Parse group_name like "DC=HCM, Region=South"
                for part in group_name.split(', '):
                    if '=' in part:
                        col, val = part.split('=', 1)
                        if col not in available_filters:
                            available_filters[col] = set()
                        available_filters[col].add(val)
            
            # Create filter UI with only available options
            st.markdown("### 🔭 Visualization Filters")
            
            # Create columns for filters (max 4 per row)
            num_filters = len(available_filters)
            num_cols = min(4, num_filters)
            col_filters = st.columns(num_cols)
            
            selected_filters = {}
            for idx, (col, values) in enumerate(available_filters.items()):
                with col_filters[idx % num_cols]:
                    sorted_values = ['All'] + sorted(list(values))
                    selected_val = st.selectbox(
                        f"🔎 {col}",
                        sorted_values,
                        key=f"filter_{col}",
                        help=f"Filter by {col}"
                    )
                    
                    if selected_val != 'All':
                        selected_filters[col] = selected_val
            
            # More filters if needed (for more than 4 columns)
            if num_filters > 4:
                remaining_filters = list(available_filters.items())[4:]
                with st.expander("➕ More Filters"):
                    more_cols = st.columns(3)
                    for idx, (col, values) in enumerate(remaining_filters):
                        with more_cols[idx % 3]:
                            sorted_values = ['All'] + sorted(list(values))
                            selected_val = st.selectbox(
                                f"🔎 {col}",
                                sorted_values,
                                key=f"filter_more_{col}",
                                help=f"Filter by {col}"
                            )
                            
                            if selected_val != 'All':
                                selected_filters[col] = selected_val
            
            # Find matching group(s) based on filters
            matching_groups = []
            for group_name in group_results.keys():
                match = True
                if selected_filters:
                    for filter_col, filter_val in selected_filters.items():
                        if f"{filter_col}={filter_val}" not in group_name:
                            match = False
                            break
                if match:
                    matching_groups.append(group_name)
            
            # Check if ALL group columns have been selected (none are "All")
            all_filters_selected = len(selected_filters) == len(available_filters)
            
            if all_filters_selected:
                # Show specific group forecast
                filter_summary = ', '.join([f"{k}={v}" for k, v in selected_filters.items()])
                st.info(f"🔍 **Active Filters:** {filter_summary}")
                
                st.markdown("---")
                
                if matching_groups:
                    # Display each matching group's forecast
                    for group_name in matching_groups:
                        result = group_results[group_name]
                        
                        st.markdown(f"### 📈 {group_name}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Best Model", result['best_model_name'].split('. ')[1])
                        col2.metric("MAPE", f"{result['best_model_result']['metrics']['MAPE']:.2f}%")
                        col3.metric("MAE", f"{result['best_model_result']['metrics']['MAE']:.2f}")
                        col4.metric("RMSE", f"{result['best_model_result']['metrics']['RMSE']:.2f}")
                        
                        # Generate and display future forecast
                        with st.spinner(f"Generating forecast for {group_name}..."):
                            # Use stored future forecast from training phase
                            future_forecast = result.get('future_forecast', np.array([]))
                            
                            if len(future_forecast) > 0:
                                # Get fitted values - use test predictions
                                test_predictions = result['best_model_result']['predictions']
                                
                                # Create fitted values that match historical length
                                full_fitted = np.concatenate([
                                    result['train_data'],     # Training portion = actual values
                                    test_predictions          # Test portion = model predictions
                                ])
                                
                                # Plot with fitted line that connects to forecast
                                fig = viz.plot_forecast(
                                    result['full_data'],
                                    full_fitted,
                                    future_forecast,
                                    title=f"Forecast for {group_name}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        if len(matching_groups) > 1:
                            st.markdown("---")
                else:
                    st.warning("⚠️ No groups match these filters. Try different combinations.")
            else:
                # Show overall forecast (aggregated across all groups)
                st.info("📊 **Viewing:** Overall/Aggregated forecast (Sum of matching groups)")
                if selected_filters:
                    filter_summary = ', '.join([f"{k}={v}" for k, v in selected_filters.items()])
                    st.info(f"🔍 **Partial Filters:** {filter_summary}")
                
                st.markdown("---")
                
                # Aggregate data from all matching groups and use their best-fit models
                aggregated_y, aggregated_future_forecast, model_info = aggregate_forecasts_from_groups(
                    matching_groups, group_results, settings, train_ratio, min_periods
                )
                
                if aggregated_y is not None and len(aggregated_y) > 0:
                    st.markdown(f"### 🔼 Overall Forecast (Sum of {len(matching_groups)} groups)")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Groups Summed", len(matching_groups))
                    col2.metric("Models Aggregated", len(model_info['models_used']))
                    
                    # Show which models are being used
                    # with st.expander("📋 Models used by each group"):
                    #     models_text = "Models breakdown:\n"
                    #     for model_name, count in sorted(model_info['models_used'].items(), key=lambda x: x[1], reverse=True):
                    #         models_text += f"- {model_name}: {count} group(s)\n"
                    #     st.text(models_text)
                    
                    # Debug: Show individual group forecasts and their sum
                    # with st.expander("🔍 Debug: Individual group forecasts (Last forecast period)"):
                    #     if 'group_forecasts' in model_info and model_info['group_forecasts']:
                    #         # Get the last period index available
                    #         first_forecast = next(iter(model_info['group_forecasts'].values()), None)
                    #         if first_forecast is not None:
                    #             last_period_idx = len(first_forecast) - 1
                    #             last_period_num = settings['forecast_periods']
                                
                    #             st.write(f"**Showing last forecast period (Period {last_period_num} ahead):**")
                    #             st.divider()
                                
                    #             for group_name, group_forecast in model_info['group_forecasts'].items():
                    #                 if len(group_forecast) > 0:
                    #                     value = group_forecast[last_period_idx]
                    #                     value_str = f"{value:.6f}"
                    #                     st.write(f"**{group_name}**: {value_str}")
                                
                    #             # Show the aggregated sum
                    #             st.divider()
                    #             agg_value = aggregated_future_forecast[last_period_idx]
                    #             agg_value_str = f"{agg_value:.6f}"
                    #             st.write(f"**Aggregated Sum (All groups):** {agg_value_str}")
                                
                    #             # Verify the sum
                    #             individual_sum = sum([
                    #                 model_info['group_forecasts'][gname][last_period_idx]
                    #                 for gname in model_info['group_forecasts'].keys()
                    #             ])
                    #             st.write(f"**Manual Sum Verification:** {individual_sum:.6f}")
                    #             if abs(individual_sum - agg_value) < 0.01:
                    #                 st.success("✅ Sum matches perfectly!")
                    #             else:
                    #                 st.warning(f"⚠️ Difference: {abs(individual_sum - agg_value):.6f}")
                    
                    # Generate fitted values for aggregated data
                    # Split aggregated data for train/test
                    test_size_agg = int(len(aggregated_y) * (1 - train_ratio / 100))
                    test_size_agg = max(1, min(test_size_agg, len(aggregated_y) - min_periods))
                    
                    train_data_agg = aggregated_y[:-test_size_agg]
                    test_data_agg = aggregated_y[-test_size_agg:]
                    
                    # For fitted values, we approximate by training the best model on aggregated train data
                    with st.spinner(f"Generating overall forecast..."):
                        test_result = settings['factory'].train_and_predict(
                            model_info['model_name'],
                            train_data_agg,
                            len(test_data_agg)
                        )
                        
                        if test_result and len(aggregated_future_forecast) > 0:
                            # Create fitted values: train portion = actual, test portion = predictions
                            test_predictions = np.maximum(test_result['predictions'], 0)
                            full_fitted = np.concatenate([
                                train_data_agg,      # Training portion = actual
                                test_predictions      # Test portion = model predictions
                            ])
                            
                            fig = viz.plot_forecast(
                                aggregated_y,
                                full_fitted,
                                aggregated_future_forecast,
                                title=f"Overall Forecast (Sum of {len(matching_groups)} groups)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 No group filters available. Select grouping columns during configuration to enable filtering.")
        
        # Download all results
        if st.button("📥 Download All Group Forecasts", type="primary"):
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual group forecasts
                for group_name, result in group_results.items():
                    # Use stored forecast from training phase
                    future_forecast = result.get('future_forecast', np.array([]))
                    
                    if len(future_forecast) > 0:
                        # Parse group name to extract column values
                        # e.g., "DC=VNH2, Product Name=Actrapid" -> {DC: VNH2, Product Name: Actrapid}
                        group_dict = {}
                        for part in group_name.split(', '):
                            if '=' in part:
                                col, val = part.split('=', 1)
                                group_dict[col] = val
                        
                        # Create period numbers
                        num_actual = len(result['full_data'])
                        num_forecast = settings['forecast_periods']
                        
                        # Build the dataframe
                        periods = list(range(1, num_actual + num_forecast + 1))
                        
                        # Create columns for group-by values
                        data_dict = {'Period': periods}
                        for col, val in group_dict.items():
                            data_dict[col] = [val] * len(periods)
                        
                        # Create Actual column (with NaN for forecast periods)
                        actual_values = list(result['full_data']) + [np.nan] * num_forecast
                        data_dict['Actual'] = actual_values
                        
                        # Create Forecast column (with NaN for actual periods)
                        forecast_values = [np.nan] * num_actual + list(future_forecast)
                        data_dict['Forecast'] = forecast_values
                        
                        # Add model name
                        data_dict['Model'] = [result['best_model_name']] * len(periods)
                        
                        forecast_data = pd.DataFrame(data_dict)
                        
                        # Clean sheet name (Excel limit: 31 chars)
                        sheet_name = group_name[:31].replace('/', '-').replace(':', '-')
                        forecast_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.download_button(
                label='📥 Download Complete Analysis',
                data=output.getvalue(),
                file_name='group_wise_forecasts.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    
    # Overall results
    elif 'overall_results' in st.session_state:
        st.markdown("---")
        st.subheader("🏆 Model Performance Ranking")
        
        results = st.session_state.overall_results
        settings = st.session_state.forecast_settings
        sorted_models = results['sorted_models']
        
        # Performance table
        performance_data = []
        for rank, (name, result) in enumerate(sorted_models, 1):
            metrics = result['metrics']
            performance_data.append({
                'Rank': rank,
                'Model': name,
                'MAPE (%)': metrics['MAPE'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R²': metrics['R²']
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        def highlight_top3(row):
            if row['Rank'] <= 3:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            perf_df.style.apply(highlight_top3, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Future forecast with best model
        st.markdown("---")
        st.subheader(f"🔮 Future Forecast ({settings['forecast_periods']} periods)")
        
        best_name, best_result = sorted_models[0]
        
        with st.spinner(f'Generating forecast with {best_name}...'):
            # Get fitted values
            fitted_values = best_result['predictions']
            full_fitted = np.concatenate([
                results['train_data'],
                fitted_values
            ])
            
            future_result = settings['factory'].train_and_predict(
                best_name,
                results['full_data'],
                settings['forecast_periods']
            )
            
            if future_result:
                future_forecast = np.maximum(future_result['predictions'], 0)
                
                fig = viz.plot_forecast(
                    results['full_data'],
                    full_fitted,
                    future_forecast,
                    title=f"Future Forecast using {best_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download
                if st.button("📥 Download Results", type="primary"):
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        perf_df.to_excel(writer, sheet_name='Performance', index=False)
                        
                        # Create combined actual + forecast dataframe
                        num_actual = len(results['full_data'])
                        num_forecast = settings['forecast_periods']
                        periods = list(range(1, num_actual + num_forecast + 1))
                        
                        # Actual values (with NaN for forecast periods)
                        actual_values = list(results['full_data']) + [np.nan] * num_forecast
                        
                        # Forecast values (with NaN for actual periods)
                        forecast_values = [np.nan] * num_actual + list(future_forecast)
                        
                        forecast_df = pd.DataFrame({
                            'Period': periods,
                            'Actual': actual_values,
                            'Forecast': forecast_values,
                            'Model': [best_name] * len(periods)
                        })
                        forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
                    
                    st.download_button(
                        label='📥 Download Analysis',
                        data=output.getvalue(),
                        file_name='forecast_analysis.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )