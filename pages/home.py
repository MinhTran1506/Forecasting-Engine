"""
Home page
"""

import streamlit as st
import numpy as np
def render(df):
    st.header("Welcome to Future1 Pro 🚀")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Advanced Multi-Model Forecasting Platform
        
        Future1 Pro is an enterprise-grade forecasting application powered by state-of-the-art 
        machine learning algorithms. Make data-driven decisions with confidence.
        
        #### ✨ Key Features
        - **17+ Forecasting Models**: From classical to cutting-edge ML algorithms
        - **Automatic Model Selection**: AI recommends the best model for your data
        - **Flexible Data Input**: Support for CSV, Excel, and various data structures
        - **Multi-Level Analysis**: Forecast at product, location, or channel level
        - **Enterprise Security**: Role-based access control and data encryption
        - **Interactive Visualizations**: Rich, interactive charts and comparisons
        - **Export Results**: Download forecasts in Excel format
        
        #### 📊 Supported Models
        
        **Classical Methods:**
        - Simple & Weighted Average
        - Simple & Weighted Moving Average
        - Linear & Seasonal Regression
        
        **Exponential Smoothing:**
        - Single Exponential Smoothing
        - Double Exponential Smoothing (Holt's)
        - Triple Exponential Smoothing (Holt-Winters)
        - Automated Exponential Smoothing
        - Adaptive Response Rate Smoothing
        - Brown's Linear Exponential Smoothing
        
        **Advanced ML:**
        - Auto-ARIMA & SARIMAX
        - Gradient Boosting
        - XGBoost
        - Facebook Prophet
        
        #### 🚀 Quick Start Guide
        
        1. **Upload Data**: Click "Upload Data File" in the sidebar
        2. **Explore Models**: Navigate through different forecasting methods
        3. **Compare Performance**: Use "Multi-Model AI" to test all models
        4. **Generate Forecasts**: Let AI select the best model automatically
        5. **Export Results**: Download your forecasts for reporting
        """)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        if df is not None:
            st.metric("Data Loaded", "✅ Active")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Columns", len(df.columns))
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Fields", len(numeric_cols))
        else:
            st.metric("Data Loaded", "⏳ Pending")
            st.info("Upload data to get started")
        
        st.markdown("---")
        st.markdown("### 🔒 Security")
        if 'user_role' in st.session_state:
            st.success(f"Role: {st.session_state.user_role.upper()}")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("- [User Guide](#)")
        st.markdown("- [API Documentation](#)")
        st.markdown("- [Video Tutorials](#)")
        st.markdown("- [Contact Support](#)")
    
    if df is not None:
        st.markdown("---")
        st.subheader("📋 Data Preview")
        
        tab1, tab2 = st.tabs(["Preview", "Statistics"])
        
        with tab1:
            st.dataframe(df.head(15), use_container_width=True)
        
        with tab2:
            st.write(df.describe())
