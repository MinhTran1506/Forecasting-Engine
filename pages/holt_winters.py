"""
Holt-Winters forecasting page
"""

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils.metrics import ForecastMetrics
from utils.visualizer import Visualizer
import io

def render(df):
    st.header("🔮 Holt-Winters Exponential Smoothing")
    
    if df is None or 'Vol' not in df.columns:
        st.warning("⚠️ Please upload data with 'Vol' column for Holt-Winters forecasting")
        st.info("💡 Or navigate to Multi-Model AI tab for flexible data input")
        return
    
    # Prepare data
    data = pd.Series(df['Vol'].values)
    
    # Calculate optimized parameters
    try:
        if len(data) >= 24:
            model = ExponentialSmoothing(
                data, trend='add', seasonal='add', 
                seasonal_periods=12, initialization_method="estimated"
            )
        else:
            model = ExponentialSmoothing(
                data, trend='add', seasonal='add', 
                seasonal_periods=3, initialization_method="estimated"
            )
        
        optimized_model = model.fit(optimized=True)
        opt_alpha = float(optimized_model.params['smoothing_level'])
        opt_beta = float(optimized_model.params['smoothing_trend'])
        opt_gamma = float(optimized_model.params['smoothing_seasonal'])
    except Exception as e:
        st.error(f"Error optimizing parameters: {e}")
        return
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Model Parameters")
    
    def reset_value():
        st.session_state.alpha = opt_alpha
        st.session_state.beta = opt_beta
        st.session_state.gamma = opt_gamma
    
    alpha = st.sidebar.slider('Alpha (Level)', 0.0, 1.0, opt_alpha, 0.01, key="alpha")
    beta = st.sidebar.slider('Beta (Trend)', 0.0, 1.0, opt_beta, 0.01, key="beta")
    gamma = st.sidebar.slider('Gamma (Seasonal)', 0.0, 1.0, opt_gamma, 0.01, key="gamma")
    periods = st.sidebar.slider('Forecast Periods', 1, 96, 36, 1, key="periods")
    
    st.sidebar.button("Reset to Optimized", on_click=reset_value)
    
    # Display parameters
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Alpha", f"{alpha:.4f}")
    col2.metric("Beta", f"{beta:.4f}")
    col3.metric("Gamma", f"{gamma:.4f}")
    col4.metric("Optimized", "✅" if (alpha==opt_alpha and beta==opt_beta and gamma==opt_gamma) else "⚙️")
    
    # Fit model and forecast
    try:
        fitted_model = model.fit(
            smoothing_level=alpha, 
            smoothing_slope=beta, 
            smoothing_seasonal=gamma
        )
        forecast = fitted_model.forecast(periods)
        
        # Calculate metrics on test set
        if len(data) >= 12:
            test = data.values[-12:]
            pred = forecast[:len(test)]
            metrics = ForecastMetrics.calculate_all(test, pred)
            
            st.subheader("📊 Model Performance")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            col2.metric("MAE", f"{metrics['MAE']:.2f}")
            col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col4.metric("MSE", f"{metrics['MSE']:.2f}")
            col5.metric("R²", f"{metrics['R²']:.4f}")
            col6.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")
        
        # Visualization
        st.subheader("📈 Forecast Visualization")
        viz = Visualizer()
        fig = viz.plot_forecast(data.values, fitted_model.fittedvalues, forecast, 
                               title="Holt-Winters Forecast")
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual analysis
        with st.expander("🔍 Residual Analysis"):
            residuals_fig = viz.plot_residuals(
                data.values, 
                fitted_model.fittedvalues
            )
            st.plotly_chart(residuals_fig, use_container_width=True)
        
        # Export results
        if st.button('📥 Download Results', type="primary"):
            # Prepare data
            time_range = pd.date_range('2020-12-01', periods=len(data), freq='MS')
            results_df = pd.DataFrame({
                'Date': time_range.strftime('%Y-%m'),
                'Actual': data.values,
                'Fitted': fitted_model.fittedvalues
            })
            
            forecast_dates = pd.date_range(
                time_range[-1] + pd.DateOffset(months=1), 
                periods=len(forecast), 
                freq='MS'
            )
            forecast_df = pd.DataFrame({
                'Date': forecast_dates.strftime('%Y-%m'),
                'Forecast': forecast
            })
            
            # Create Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, sheet_name='Historical', index=False)
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            
            st.download_button(
                label='📥 Download Excel',
                data=output.getvalue(),
                file_name='holt_winters_forecast.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    
    except Exception as e:
        st.error(f"Error in forecasting: {e}")