import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from auth.authenticator import Authenticator
from pages import home, holt_winters, auto_future, multi_model
from utils.data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Future1 Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Authentication
auth = Authenticator()

if not st.session_state.authenticated:
    # Hide sidebar on login page
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    auth.login_page()
else:
    # Header with logout
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<p style='font-family: impact; font-size: 80px; font-style: italic; text-align: center; margin-bottom: 0px;'>Future1</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family: impact; font-size: 14px; text-align: center; margin-top: 0px;'>Pro Edition | User: {st.session_state.username}</p>", unsafe_allow_html=True)
    with col2:
        if st.button("Logout", use_container_width=True):
            auth.logout()
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    # Sidebar
    st.sidebar.header("📊 Data Configuration")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Data File", 
        type=["csv", "xlsx"],
        help="Upload your time series data in CSV or Excel format"
    )
    
    # Load data
    processor = DataProcessor()
    if uploaded_file is not None:
        df = processor.load_data(uploaded_file)
        st.session_state.data = df
    else:
        # Try to load default data
        try:
            df = processor.load_default_data()
            st.session_state.data = df
        except:
            df = None
            st.session_state.data = None
    
    # Display data info in sidebar
    if df is not None:
        st.sidebar.success(f"✅ Data loaded: {len(df)} rows")
        st.sidebar.info(f"📋 Columns: {len(df.columns)}")
    
    # Main tabs
    tab_titles = ["🏠 Home", "🔮 Holt-Winters", "🤖 Auto-Future", "🎯 Multi-Model AI"]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        home.render(df)
    
    with tabs[1]:
        holt_winters.render(df)
    
    with tabs[2]:
        auto_future.render(df)
    
    with tabs[3]:
        multi_model.render(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Future1 Pro v2.0**")
    st.sidebar.markdown("© 2024 - Advanced ML Forecasting")
    st.sidebar.markdown("[📖 Documentation](#) | [🐛 Report Bug](#)")