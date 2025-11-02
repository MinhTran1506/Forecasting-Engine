"""
Multi-model comparison page with all 17+ models
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.metrics import ForecastMetrics
from utils.visualizer import Visualizer
from models.all_models import ModelFactory
import io

def render(df):
    st.header("🎯 Multi-Model AI Comparison")
    st.markdown("**Train 17+ models simultaneously and let AI recommend the best one**")
    
    if df is None:
        st.warning("⚠️ Please upload data to use Multi-Model Comparison")
        return
    
    processor = DataProcessor()
    
    # Configuration
    st.subheader("⚙️ Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detect and select date column
        all_cols = df.columns.tolist()
        date_col = processor.detect_date_column(df)
        date_col_idx = all_cols.index(date_col) if date_col else 0
        
        selected_date = st.selectbox(
            "📅 Date Column",
            all_cols,
            index=date_col_idx,
            help="Select the column containing dates"
        )
    
    with col2:
        # Select value column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox(
            "📊 Value Column (to forecast)",
            numeric_cols,
            help="Select the numeric column you want to forecast"
        )
    
    # Aggregation options
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != selected_date]
    
    if categorical_cols:
        agg_level = st.multiselect(
            "🎚️ Group By (Optional)",
            categorical_cols,
            help="Select columns to group by for multi-level forecasting"
        )
    else:
        agg_level = []
    
    # Forecast configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_periods = st.slider("🔮 Forecast Periods", 1, 60, 12)
    with col2:
        # Intelligent default based on data length
        max_test = min(12, len(df) // 4)  # Max 25% of data
        default_test = min(6, max_test)  # Default 6 months
        test_size = st.slider("📊 Test Set Size", 3, max_test, default_test,
                              help=f"Recommended: 6-12 months. Using more than 25% reduces training data.")
    with col3:
        confidence_level = st.slider("📈 Confidence Level", 80, 99, 95)
    
    # Data preprocessing options
    with st.expander("🔧 Advanced Preprocessing Options"):
        remove_outliers = st.checkbox("Remove Outliers", value=True,
                                     help="Remove extreme values that may distort forecasts")
        
        if remove_outliers:
            outlier_std = st.slider("Outlier Threshold (Std Dev)", 1.5, 4.0, 3.0, 0.5,
                                   help="Values beyond this many standard deviations are removed")
        
        smooth_data = st.checkbox("Smooth Data", value=False,
                                 help="Apply moving average smoothing to reduce noise")
        
        if smooth_data:
            smooth_window = st.slider("Smoothing Window", 3, 12, 3,
                                     help="Larger windows = more smoothing")
        
        log_transform = st.checkbox("Log Transform", value=False,
                                   help="Useful for data with exponential growth")
    
    # Model selection
    st.markdown("### 🎯 Model Selection")
    model_selection = st.radio(
        "Choose models to train:",
        ["All Models (17)", "Fast Models Only (10)", "Best Performers (Top 5)", "Custom Selection"],
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
        selected_models = st.multiselect("Select models:", available_models, default=available_models[:5])
    
    # Validate data
    validation_issues = processor.validate_time_series(df, selected_date, value_col)
    if validation_issues:
        with st.expander("⚠️ Data Validation Warnings"):
            for issue in validation_issues:
                st.warning(issue)
    
    # Run forecast button
    if st.button("🚀 Train All Models & Generate Forecasts", type="primary", use_container_width=True):
        with st.spinner('🔄 Preparing data and training 17+ models... This may take a moment.'):
            try:
                # Prepare time series
                ts_data = processor.prepare_time_series(
                    df, selected_date, value_col, 
                    agg_level if agg_level else None
                )
                
                if ts_data is None:
                    return
                
                y = ts_data[value_col].values
                dates = ts_data[selected_date].values
                
                # Apply preprocessing
                original_y = y.copy()
                preprocessing_applied = []
                
                if remove_outliers:
                    y, mask = processor.remove_outliers(y, outlier_std)
                    dates = dates[mask]
                    preprocessing_applied.append(f"Outlier Removal (±{outlier_std}σ)")
                
                if smooth_data:
                    y = processor.smooth_series(y, smooth_window)
                    preprocessing_applied.append(f"Smoothing (window={smooth_window})")
                
                if log_transform:
                    y = processor.log_transform_series(y)
                    preprocessing_applied.append("Log Transform")
                
                if preprocessing_applied:
                    st.info(f"✨ Applied: {', '.join(preprocessing_applied)}")
                
                # Show data quality metrics
                with st.expander("📊 Data Quality Analysis"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{np.mean(y):.2f}")
                    col2.metric("Std Dev", f"{np.std(y):.2f}")
                    col3.metric("CV", f"{np.std(y)/np.mean(y)*100:.1f}%")
                    col4.metric("Min/Max", f"{np.min(y):.0f} / {np.max(y):.0f}")
                
                # Check minimum data length
                min_required = test_size + 24  # Need at least 24 training points
                if len(y) < min_required:
                    st.error(f"❌ Not enough data. Need at least {min_required} observations, got {len(y)}.")
                    st.info("💡 Try reducing test set size or upload more historical data.")
                    return
                
                # Split data
                train_data = y[:-test_size]
                test_data = y[-test_size:]
                
                st.success(f"✅ Data prepared: {len(train_data)} training, {len(test_data)} test observations")
                
                # Show train/test split info
                train_ratio = len(train_data) / len(y) * 100
                if train_ratio < 70:
                    st.warning(f"⚠️ Only {train_ratio:.1f}% of data used for training. Consider reducing test size.")
                else:
                    st.info(f"📊 Train/Test split: {train_ratio:.1f}% / {100-train_ratio:.1f}%")
                
                # Initialize model factory
                factory = ModelFactory()
                
                # Determine which models to train
                if model_selection == "All Models (17)":
                    model_names = factory.get_all_model_names()
                elif model_selection == "Fast Models Only (10)":
                    model_names = [m for m in factory.get_all_model_names() 
                                 if not any(x in m for x in ['ARIMA', 'SARIMAX', 'Prophet', 'Gradient', 'XGBoost'])]
                elif model_selection == "Best Performers (Top 5)":
                    model_names = [
                        '5. Linear Regression',
                        '6. Seasonal Linear Regression', 
                        '10. Automated Exp Smoothing',
                        '9. Triple Exponential Smoothing',
                        '17. Prophet'
                    ]
                else:  # Custom Selection
                    # Map friendly names to internal names
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
                
                st.info(f"🎯 Training {len(model_names)} models...")
                
                # Train all models
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = {}
                
                for idx, model_name in enumerate(model_names):
                    status_text.text(f"Training {model_name}... ({idx+1}/{len(model_names)})")
                    
                    try:
                        result = factory.train_and_predict(
                            model_name, train_data, len(test_data)
                        )
                        
                        if result is not None:
                            metrics = ForecastMetrics.calculate_all(test_data, result['predictions'])
                            all_results[model_name] = {
                                'predictions': result['predictions'],
                                'metrics': metrics,
                                'model': result.get('model')
                            }
                    except Exception as e:
                        st.warning(f"⚠️ {model_name} failed: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(model_names))
                
                progress_bar.empty()
                status_text.empty()
                
                if not all_results:
                    st.error("No models trained successfully. Please check your data.")
                    return
                
                # Sort by MAPE
                sorted_models = sorted(
                    all_results.items(),
                    key=lambda x: x[1]['metrics']['MAPE']
                )
                
                st.success(f"🎉 Successfully trained {len(all_results)} models!")
                
                # Display results
                st.markdown("---")
                st.subheader("🏆 Model Performance Ranking")
                
                # Performance interpretation
                best_mape = sorted_models[0][1]['metrics']['MAPE']
                
                if best_mape < 10:
                    accuracy_level = "🟢 Excellent"
                    accuracy_desc = "Highly accurate forecasts suitable for critical decisions"
                elif best_mape < 20:
                    accuracy_level = "🟡 Good"
                    accuracy_desc = "Reliable forecasts for planning purposes"
                elif best_mape < 30:
                    accuracy_level = "🟠 Acceptable"
                    accuracy_desc = "Moderate accuracy - use with caution"
                elif best_mape < 50:
                    accuracy_level = "🔴 Poor"
                    accuracy_desc = "Low accuracy - consider data improvements"
                else:
                    accuracy_level = "⛔ Very Poor"
                    accuracy_desc = "Not recommended for decision making"
                
                st.info(f"**Best Model Accuracy: {accuracy_level}** - {accuracy_desc}")
                
                # Improvement recommendations
                if best_mape > 30:
                    with st.expander("💡 Recommendations to Improve Accuracy", expanded=True):
                        st.markdown("""
                        ### 🔧 Actions to Take:
                        
                        **1. Reduce Test Set Size** ⚠️ PRIORITY
                        - Current: {test_size} months
                        - Recommended: 6-12 months maximum
                        - More training data = better models
                        
                        **2. Check Data Quality**
                        - Remove outliers (enable in Advanced Options)
                        - Look for missing values or zeros
                        - Verify data aggregation is correct
                        
                        **3. Add More Historical Data**
                        - Current: {len_y} periods
                        - Recommended: 36+ months for seasonal patterns
                        - More data helps capture trends
                        
                        **4. Try Preprocessing**
                        - Enable "Remove Outliers" option
                        - Try "Smooth Data" for noisy data
                        - Consider "Log Transform" for exponential growth
                        
                        **5. Consider External Factors**
                        - Your data may be influenced by:
                          - Promotions/marketing campaigns
                          - Holidays and seasons
                          - Economic conditions
                          - Competitor actions
                        - Simple models can't account for these!
                        
                        **6. Try Ensemble Methods**
                        - Average predictions from top 3 models
                        - Often more accurate than single model
                        
                        **7. Use Different Aggregation**
                        - Try forecasting at different levels
                        - Product-level vs. Total sales
                        - Monthly vs. Weekly data
                        """.format(test_size=test_size, len_y=len(y)))
                
                # Create performance table
                performance_data = []
                for rank, (name, result) in enumerate(sorted_models, 1):
                    metrics = result['metrics']
                    performance_data.append({
                        'Rank': rank,
                        'Model': name,
                        'MAPE (%)': metrics['MAPE'],
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'R²': metrics['R²'],
                        'SMAPE (%)': metrics['SMAPE']
                    })
                
                perf_df = pd.DataFrame(performance_data)
                
                # Highlight top 3
                def highlight_top3(row):
                    if row['Rank'] <= 3:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    perf_df.style.apply(highlight_top3, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Best model recommendation
                best_name, best_result = sorted_models[0]
                st.success(f"🎯 **Recommended Model**: {best_name} (MAPE: {best_result['metrics']['MAPE']:.2f}%)")
                
                # Ensemble option
                if len(sorted_models) >= 3:
                    with st.expander("🎲 Ensemble Forecast (Advanced)", expanded=bool(best_mape > 30)):
                        st.markdown("""
                        **Ensemble forecasting** combines predictions from multiple models, 
                        often achieving better accuracy than any single model.
                        """)
                        
                        use_ensemble = st.checkbox("Use Ensemble Forecast", value=False)
                        
                        if use_ensemble:
                            ensemble_size = st.slider("Number of models to combine", 2, min(5, len(sorted_models)), 3)
                            
                            # Calculate ensemble prediction
                            ensemble_pred = np.mean([
                                result['predictions'] 
                                for _, result in sorted_models[:ensemble_size]
                            ], axis=0)
                            
                            # Calculate ensemble metrics
                            ensemble_metrics = ForecastMetrics.calculate_all(test_data, ensemble_pred)
                            
                            st.metric("Ensemble MAPE", f"{ensemble_metrics['MAPE']:.2f}%")
                            
                            improvement = best_mape - ensemble_metrics['MAPE']
                            if improvement > 0:
                                st.success(f"✅ Ensemble is {improvement:.2f}% better than best single model!")
                            else:
                                st.info(f"ℹ️ Best single model performs better by {-improvement:.2f}%")
                            
                            # Show which models are in ensemble
                            st.write(f"**Models in ensemble (Top {ensemble_size}):**")
                            for i, (name, _) in enumerate(sorted_models[:ensemble_size], 1):
                                st.write(f"{i}. {name}")
                
                # Top 3 comparison
                st.markdown("---")
                st.subheader("📊 Top 3 Models Comparison")
                
                viz = Visualizer()
                top3_predictions = {
                    f"{name} ({result['metrics']['MAPE']:.2f}%)": result['predictions']
                    for name, result in sorted_models[:3]
                }
                
                fig = viz.plot_model_comparison(test_data, top3_predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics for top 3
                with st.expander("📈 Detailed Metrics - Top 3 Models"):
                    cols = st.columns(3)
                    for idx, (name, result) in enumerate(sorted_models[:3]):
                        with cols[idx]:
                            st.markdown(f"**{name}**")
                            metrics = result['metrics']
                            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                            st.metric("R²", f"{metrics['R²']:.4f}")
                
                # Generate future forecast with best model
                st.markdown("---")
                st.subheader(f"🔮 Future Forecast ({forecast_periods} periods ahead)")
                
                with st.spinner(f'Generating forecast with {best_name}...'):
                    try:
                        # Retrain on full data
                        future_result = factory.train_and_predict(
                            best_name, y, forecast_periods
                        )
                        
                        if future_result:
                            future_forecast = future_result['predictions']
                            
                            # Plot future forecast
                            fig_future = viz.plot_forecast(
                                y, None, future_forecast,
                                title=f"Future Forecast using {best_name}"
                            )
                            st.plotly_chart(fig_future, use_container_width=True)
                            
                            # Prepare export data
                            if st.button("📥 Download All Results", type="primary"):
                                output = io.BytesIO()
                                
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    # Performance comparison
                                    perf_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
                                    
                                    # Historical data
                                    hist_df = pd.DataFrame({
                                        'Date': dates,
                                        'Actual': y
                                    })
                                    hist_df.to_excel(writer, sheet_name='Historical_Data', index=False)
                                    
                                    # Future forecast
                                    last_date = pd.to_datetime(dates[-1])
                                    future_dates = pd.date_range(
                                        last_date + pd.DateOffset(months=1),
                                        periods=forecast_periods,
                                        freq='MS'
                                    )
                                    future_df = pd.DataFrame({
                                        'Date': future_dates.strftime('%Y-%m'),
                                        'Forecast': future_forecast,
                                        'Model': best_name
                                    })
                                    future_df.to_excel(writer, sheet_name='Future_Forecast', index=False)
                                    
                                    # Top 3 model predictions
                                    top3_df = pd.DataFrame({
                                        name: result['predictions']
                                        for name, result in sorted_models[:3]
                                    })
                                    top3_df.to_excel(writer, sheet_name='Top3_Predictions', index=False)
                                
                                st.download_button(
                                    label='📥 Download Complete Analysis',
                                    data=output.getvalue(),
                                    file_name='multi_model_forecast_analysis.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )
                    
                    except Exception as e:
                        st.error(f"Error generating future forecast: {e}")
            
            except Exception as e:
                st.error(f"Error in multi-model analysis: {e}")
                st.exception(e)