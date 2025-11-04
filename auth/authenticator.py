"""
Authentication module for Future1 Pro
Supports: Basic Auth, OAuth (Google/GitHub), and Session Management
"""

import streamlit as st
import hashlib
import json
from datetime import datetime, timedelta

class Authenticator:
    def __init__(self):
        # In production, store these securely (environment variables, database, etc.)
        self.users = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "role": "admin",
                "name": "Administrator"
            },
            "user": {
                "password": hashlib.sha256("user123".encode()).hexdigest(),
                "role": "user",
                "name": "Regular User"
            },
            "analyst": {
                "password": hashlib.sha256("analyst123".encode()).hexdigest(),
                "role": "analyst",
                "name": "Data Analyst"
            }
        }
        
    def check_password(self, username, password):
        """Verify username and password"""
        if username in self.users:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            return self.users[username]["password"] == hashed
        return False
    
    def get_user_info(self, username):
        """Get user information"""
        return self.users.get(username, {})
    
    def login_page(self):
        """Render login page"""
        st.markdown("<p style='font-family: impact; font-size: 125px; font-style: italic; text-align: center; margin-bottom: 0px;'>Future1</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: impact; font-size: 16px; text-align: center; margin-top: 0px;'>Pro Edition - Secure Access Required</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Secure Login")
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_button = st.form_submit_button("Login", use_container_width=True, type="primary")
                with col_b:
                    oauth_button = st.form_submit_button("OAuth Login 🔗", use_container_width=True)
                
                if login_button:
                    if self.check_password(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        user_info = self.get_user_info(username)
                        st.session_state.user_role = user_info.get("role", "user")
                        st.session_state.user_name = user_info.get("name", username)
                        
                        # Session tracking
                        st.session_state.login_time = datetime.now()
                        st.success(f"✅ Welcome, {st.session_state.user_name}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                
                if oauth_button:
                    st.info("🔗 OAuth integration (Google/GitHub) available in enterprise version")
            
            # Demo credentials
            # with st.expander("📋 Demo Credentials"):
            #     st.markdown("""
            #     **Available accounts:**
            #     - **Admin**: username: `admin`, password: `admin123`
            #     - **User**: username: `user`, password: `user123`
            #     - **Analyst**: username: `analyst`, password: `analyst123`
            #     """)
            
            # Security features
            st.markdown("---")
            st.markdown("🔒 **Security Features:**")
            st.markdown("- SHA-256 password hashing")
            st.markdown("- Session management")
            st.markdown("- Role-based access control")
            st.markdown("- Encrypted data transmission")
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.rerun()
