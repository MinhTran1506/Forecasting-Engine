# Future1 Pro - Advanced Multi-Model Forecasting Platform

## 🚀 Features

- **17+ Forecasting Models**: Classical to cutting-edge ML algorithms
- **Automatic Model Selection**: AI-powered best model recommendation
- **Flexible Data Input**: Support for CSV and Excel files
- **Multi-Level Analysis**: Forecast at product, location, or channel level
- **Enterprise Security**: Multi-layer authentication system
- **Interactive Visualizations**: Rich, interactive Plotly charts
- **Export Results**: Download forecasts in Excel format

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/future1-pro.git
cd future1-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Running the App

```bash
streamlit run app.py
```

## 📁 Project Structure

```
future1_pro/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── config/
│   └── settings.py            # App settings
├── auth/
│   └── authenticator.py       # Authentication
├── models/
│   └── all_models.py          # All forecasting models
├── utils/
│   ├── data_processor.py      # Data processing
│   ├── metrics.py             # Forecast metrics
│   └── visualizer.py          # Visualizations
└── pages/
    ├── home.py                # Home page
    ├── holt_winters.py        # Holt-Winters
    ├── auto_future.py         # Prophet
    └── multi_model.py         # Multi-model comparison
```

## 🔐 Security Features

- SHA-256 password hashing
- Session management
- Role-based access control
- XSRF protection
- Secure file uploads

## 📊 Supported Models

### Classical Methods
1. Simple Average
2. Weighted Average
3. Simple Moving Average
4. Weighted Moving Average
5. Linear Regression
6. Seasonal Linear Regression

### Exponential Smoothing
7. Single Exponential Smoothing
8. Double Exponential Smoothing (Holt's)
9. Triple Exponential Smoothing (Holt-Winters)
10. Automated Exponential Smoothing
11. Adaptive Response Rate Smoothing
12. Brown's Linear Exponential Smoothing

### Advanced ML
13. Auto-ARIMA
14. SARIMAX
15. Gradient Boosting
16. XGBoost-like variant
17. Facebook Prophet

## 🎯 Usage

1. **Login**: Use demo credentials (admin/admin123)
2. **Upload Data**: CSV or Excel file with time series data
3. **Configure**: Select date and value columns
4. **Choose Method**:
   - Holt-Winters: Manual parameter tuning
   - Auto-Future: Automated Prophet forecasting
   - Multi-Model AI: Compare all models automatically
5. **Analyze**: View performance metrics and visualizations
6. **Export**: Download results in Excel format

## 📈 Data Format

Your data should include:
- **Date column**: Any date/time field
- **Value column**: Numeric values to forecast
- **Optional**: Additional columns for grouping

Example:
```csv
Date,Product,Region,Sales
2023-01-01,ProductA,North,1000
2023-02-01,ProductA,North,1200
...
```

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Forecast periods range
- Model parameters
- UI settings
- Security settings

## 🚀 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy!

### Docker (Coming Soon)

```bash
docker build -t future1-pro .
docker run -p 8501:8501 future1-pro
```

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md

## 📧 Support

For issues and questions:
- GitHub Issues
- Email: minhtranquang824@gmail.com
- Documentation: docs.future1pro.com

## 🎓 Credits

Developed with:
- Streamlit
- Scikit-learn
- Statsmodels
- Prophet
- Plotly

---

**Future1 Pro v2.0** - See the future in 1 click! 🔮