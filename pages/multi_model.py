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

def render(df):
    st.header("🎯 Multi-Model AI Comparison")
    st.markdown("**Train 17+ models for each group and let AI recommend the best one per group**")
    
    if df is None:
        st.warning("⚠️ Please upload data to use Multi-Model Comparison")
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
    st.markdown("### 🎚️ Forecast Granularity (Group-wise Training)")
    st.info("💡 **How it works:** Each model will be trained separately for each unique combination of groups. For example, if you select 'DC' and have 7 DCs with 10 models, the system will train 70 models total (10 models × 7 DCs) and select the best model for each DC.")
    
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
            max_value=60,
            value=12,
            step=1,
            help="Number of periods to forecast ahead"
        )
    
    with col2:
        train_ratio = st.number_input(
            "📊 Train Ratio (%)",
            min_value=50,
            max_value=90,
            value=70,
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
            min_value=6,
            max_value=24,
            value=12,
            step=1,
            help="Minimum data points required per group"
        )
    
    # ========== STEP 4: PREPROCESSING OPTIONS ==========
    with st.expander("🔧 Advanced Preprocessing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_outliers = st.checkbox(
                "Remove Outliers",
                value=True,
                help="Remove extreme values that may distort forecasts"
            )
            
            if remove_outliers:
                outlier_std = st.slider(
                    "Outlier Threshold (Std Dev)",
                    1.5, 4.0, 3.0, 0.5,
                    help="Values beyond this many standard deviations are removed"
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
    st.markdown("### 🎯 Model Selection")
    model_selection = st.radio(
        "Choose models to train:",
        ["All Models (17)", "Fast Models Only (10)", "Best Performers (5)", "Custom Selection"],
        horizontal=True
    )
    
    if model_selection == "Custom Selection":
        available_models = [
            "Simple Average", "Weighted Average", "Simple Moving Average",
            "Weighted Moving Average", "Linear Regression", "Seasonal Linear Regression",
            "Single Exp Smoothing", "Double Exp Smoothing", "Triple Exp Smoothing",
            "Automated Exp Smoothing", "Adaptive Response Rate", "Browns Linear",
            "Auto-ARIMA", "SARIMAX", "Gradient Boosting", "XGBoost-like", "Prophet"
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
                elif model_selection == "Best Performers (5)":
                    model_names = [
                        '5. Linear Regression',
                        '6. Seasonal Linear Regression',
                        '10. Automated Exp Smoothing',
                        '9. Triple Exponential Smoothing',
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
                        'XGBoost-like': '16. XGBoost-like (GB variant)',
                        'Prophet': '17. Prophet'
                    }
                    model_names = [name_map[m] for m in selected_models if m in name_map]
                
                # ========== GROUP-WISE TRAINING ==========
                if group_cols:
                    # Get unique groups
                    group_combinations = agg_df[group_cols].drop_duplicates().reset_index(drop=True)
                    st.info(f"🔄 Training {len(model_names)} models for {len(group_combinations)} groups = {len(model_names) * len(group_combinations)} total models")
                    
                    all_group_results = {}
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
                            st.warning(f"⚠️ Skipping {group_name}: only {len(group_data)} periods (minimum: {min_periods})")
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
                            all_group_results[group_name] = {
                                'best_model_name': best_model[0],
                                'best_model_result': best_model[1],
                                'all_results': group_results,
                                'train_data': train_data,
                                'test_data': test_data,
                                'full_data': y,
                                'group_filter': group_row.to_dict()
                            }
                        
                        progress_bar.progress((group_idx + 1) / total_iterations)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
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
            st.markdown("### 📊 Visualization Filters")
            
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
                st.info(f"📊 Showing: {len(matching_groups)} matching group(s)")
                
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
                            future_result = settings['factory'].train_and_predict(
                                result['best_model_name'],
                                result['full_data'],
                                settings['forecast_periods']
                            )
                            
                            if future_result:
                                future_forecast = np.maximum(future_result['predictions'], 0)
                                
                                fig = viz.plot_forecast(
                                    result['full_data'],
                                    None,
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
                st.info("📊 **Viewing:** Overall/Aggregated forecast")
                if selected_filters:
                    filter_summary = ', '.join([f"{k}={v}" for k, v in selected_filters.items()])
                    st.info(f"🔍 **Partial Filters:** {filter_summary}")
                st.info("👆 **Select all filter options to view specific group forecasts**")
                
                st.markdown("---")
                
                # Aggregate data from all matching groups
                all_periods = []
                all_values = []
                
                for group_name in matching_groups:
                    result = group_results[group_name]
                    all_values.extend(result['full_data'])
                    all_periods.extend(range(len(result['full_data'])))
                
                if all_values:
                    # Aggregate by period (sum or average)
                    period_data = {}
                    for period, value in zip(all_periods, all_values):
                        if period not in period_data:
                            period_data[period] = []
                        period_data[period].append(value)
                    
                    # Average values for each period
                    aggregated_y = np.array([np.mean(period_data[p]) for p in sorted(period_data.keys())])
                    
                    st.markdown(f"### 📈 Overall Forecast (Aggregated from {len(matching_groups)} groups)")
                    
                    # Find best model from all group results (by average MAPE)
                    model_performance = {}
                    for group_name in matching_groups:
                        result = group_results[group_name]
                        model_name = result['best_model_name']
                        mape = result['best_model_result']['metrics']['MAPE']
                        
                        if model_name not in model_performance:
                            model_performance[model_name] = []
                        model_performance[model_name].append(mape)
                    
                    # Get model with best average performance
                    best_overall_model = min(model_performance.items(), 
                                            key=lambda x: np.mean(x[1]))[0]
                    avg_mape = np.mean(model_performance[best_overall_model])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Model", best_overall_model.split('. ')[1])
                    col2.metric("Avg MAPE", f"{avg_mape:.2f}%")
                    col3.metric("Groups Aggregated", len(matching_groups))
                    
                    # Generate overall forecast
                    with st.spinner(f"Generating overall forecast..."):
                        future_result = settings['factory'].train_and_predict(
                            best_overall_model,
                            aggregated_y,
                            settings['forecast_periods']
                        )
                        
                        if future_result:
                            future_forecast = np.maximum(future_result['predictions'], 0)
                            
                            fig = viz.plot_forecast(
                                aggregated_y,
                                None,
                                future_forecast,
                                title=f"Overall Forecast (Aggregated)"
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
                    future_result = settings['factory'].train_and_predict(
                        result['best_model_name'],
                        result['full_data'],
                        settings['forecast_periods']
                    )
                    
                    if future_result:
                        forecast_data = pd.DataFrame({
                            'Period': range(1, settings['forecast_periods'] + 1),
                            'Forecast': np.maximum(future_result['predictions'], 0),
                            'Model': result['best_model_name']
                        })
                        
                        # Clean sheet name (Excel limit: 31 chars)
                        sheet_name = group_name[:31].replace('/', '-')
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
            future_result = settings['factory'].train_and_predict(
                best_name,
                results['full_data'],
                settings['forecast_periods']
            )
            
            if future_result:
                future_forecast = np.maximum(future_result['predictions'], 0)
                
                fig = viz.plot_forecast(
                    results['full_data'],
                    None,
                    future_forecast,
                    title=f"Future Forecast using {best_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download
                if st.button("📥 Download Results", type="primary"):
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        perf_df.to_excel(writer, sheet_name='Performance', index=False)
                        
                        forecast_df = pd.DataFrame({
                            'Period': range(1, settings['forecast_periods'] + 1),
                            'Forecast': future_forecast,
                            'Model': best_name
                        })
                        forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
                    
                    st.download_button(
                        label='📥 Download Analysis',
                        data=output.getvalue(),
                        file_name='forecast_analysis.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )