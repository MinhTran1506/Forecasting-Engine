"""
Prophet (Auto-Future) forecasting page
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from utils.metrics import ForecastMetrics
import plotly.graph_objects as go
import io

def render(df):
    st.header("🤖 Auto-Future (Prophet)")
    
    if df is None or 'Vol' not in df.columns:
        st.warning("⚠️ Please upload data with 'Vol' column")
        st.info("💡 Or navigate to Multi-Model AI tab for flexible data input")
        return
    
    # Prepare Prophet data
    prophet_df = pd.DataFrame({
        'y': df['Vol'].values
    })
    
    start = '2020-12-01'
    start = pd.to_datetime(start)
    month_offset = pd.DateOffset(months=len(prophet_df)-1)
    end = start + month_offset
    
    prophet_df['ds'] = pd.date_range(start, end, freq='MS')
    prophet_df = prophet_df[['ds', 'y']]
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Prophet Configuration")
    forecast_periods = st.sidebar.slider('Forecast Periods', 1, 60, 36, 1)
    yearly_seasonality = st.sidebar.checkbox('Yearly Seasonality', value=True)
    weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=False)
    daily_seasonality = st.sidebar.checkbox('Daily Seasonality', value=False)
    
    changepoint_prior_scale = st.sidebar.slider(
        'Trend Flexibility', 0.001, 0.5, 0.05, 0.001,
        help="Higher values = more flexible trend"
    )
    
    seasonality_prior_scale = st.sidebar.slider(
        'Seasonality Strength', 0.01, 10.0, 10.0, 0.1,
        help="Higher values = stronger seasonality"
    )
    
    # Train model
    with st.spinner('Training Prophet model...'):
        try:
            model = Prophet(
                interval_width=0.95,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            
            model.fit(prophet_df)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
            forecast = model.predict(future)
            
            # Calculate metrics
            test_data = prophet_df['y'].values
            pred_data = forecast['yhat'].values[:len(test_data)]
            
            if len(test_data) >= 12:
                test_subset = test_data[-12:]
                pred_subset = pred_data[-12:]
                metrics = ForecastMetrics.calculate_all(test_subset, pred_subset)
                
                st.subheader("📊 Model Performance")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                col2.metric("MAE", f"{metrics['MAE']:.2f}")
                col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
                col4.metric("MSE", f"{metrics['MSE']:.2f}")
                col5.metric("R²", f"{metrics['R²']:.4f}")
                col6.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")
            
            # Visualization
            st.subheader("📈 Prophet Forecast")
            
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['y'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#F18F01', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='95% Confidence',
                fillcolor='rgba(241, 143, 1, 0.2)'
            ))
            
            fig.update_layout(
                title='Prophet Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Components
            with st.expander("📊 Forecast Components"):
                st.write("Prophet automatically detects trend and seasonality patterns:")
                
                # Trend
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['trend'],
                    mode='lines',
                    name='Trend'
                ))
                fig_trend.update_layout(title='Trend Component', height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Yearly seasonality (if enabled)
                if yearly_seasonality and 'yearly' in forecast.columns:
                    fig_yearly = go.Figure()
                    fig_yearly.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yearly'],
                        mode='lines',
                        name='Yearly'
                    ))
                    fig_yearly.update_layout(title='Yearly Seasonality', height=300)
                    st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Download results
            if st.button('📥 Download Results', type="primary"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Historical + Forecast
                    export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    export_df['actual'] = np.nan
                    export_df.loc[:len(prophet_df)-1, 'actual'] = prophet_df['y'].values
                    export_df['ds'] = export_df['ds'].dt.strftime('%Y-%m')
                    export_df.columns = ['Date', 'Forecast', 'Lower_95', 'Upper_95', 'Actual']
                    export_df.to_excel(writer, sheet_name='Forecast', index=False)
                    
                    # Components
                    components_df = forecast[['ds', 'trend']].copy()
                    if 'yearly' in forecast.columns:
                        components_df['yearly'] = forecast['yearly']
                    components_df['ds'] = components_df['ds'].dt.strftime('%Y-%m')
                    components_df.to_excel(writer, sheet_name='Components', index=False)
                
                st.download_button(
                    label='📥 Download Excel',
                    data=output.getvalue(),
                    file_name='prophet_forecast.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        
        except Exception as e:
            st.error(f"Error in Prophet forecasting: {e}")
            st.info("Try adjusting the parameters or check your data format")