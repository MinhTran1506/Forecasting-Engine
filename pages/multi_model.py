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
        test_size = st.slider("📊 Test Set Size", 3, 24, 12)
    with col3:
        confidence_level = st.slider("📈 Confidence Level", 80, 99, 95)
    
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
                print(f"Debug: ts_data={ts_data}")
                y = ts_data[value_col].values
                print(f"Debug: y={y}")
                dates = ts_data[selected_date].values
                
                # Check minimum data length
                if len(y) < test_size + 12:
                    st.error(f"Not enough data. Need at least {test_size + 12} observations. Or choose another Date Column with full date format (Contains year, month, date)")
                    return
                
                # Split data
                train_data = y[:-test_size]
                test_data = y[-test_size:]
                
                st.success(f"✅ Data prepared: {len(train_data)} training, {len(test_data)} test observations")
                
                # Initialize model factory
                factory = ModelFactory()
                
                # Train all models
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = {}
                model_names = factory.get_all_model_names()
                
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