"""
Visualization utilities
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
class Visualizer:
    @staticmethod
    def plot_forecast(actual, fitted, forecast, title="Forecast"):
        """Plot actual, fitted, and forecast values"""
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ))
        
        # Fitted
        if fitted is not None:
            fig.add_trace(go.Scatter(
                y=fitted,
                mode='lines',
                name='Fitted',
                line=dict(color='#A23B72', width=2, dash='dot')
            ))
        
        # Forecast
        if forecast is not None:
            forecast_x = list(range(len(actual)-1, len(actual) + len(forecast) - 1))
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#F18F01', width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Period',
            yaxis_title='Value',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(actual, predictions_dict):
        """Plot multiple model predictions"""
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=3),
            marker=dict(size=8)
        ))
        
        # Model predictions
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i, (name, pred) in enumerate(predictions_dict.items()):
            fig.add_trace(go.Scatter(
                y=pred,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Period',
            yaxis_title='Value',
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_residuals(actual, predicted):
        """Plot residual analysis"""
        residuals = np.array(actual) - np.array(predicted)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals Distribution')
        )
        
        # Residuals over time
        fig.add_trace(
            go.Scatter(y=residuals, mode='lines+markers', name='Residuals',
                      line=dict(color='#E74C3C')),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribution', 
                        marker_color='#3498DB', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        return fig
