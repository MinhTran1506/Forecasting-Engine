"""
Visualization utilities - Updated with connected fitted line
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

class Visualizer:
    @staticmethod
    def _thin_ticks(x_labels):
        """
        Select a readable subset of x-axis tick positions for the given labels.

        Stride by total count n = len(x_labels):
          n <= 24 -> stride 1 (all), 25..60 -> stride 3, n > 60 -> stride 6.
        The last index (n-1) is always included; it is never duplicated if the
        stride already lands on it.

        Returns (tickvals, ticktext): integer indices into x_labels and the
        matching label strings.
        """
        n = len(x_labels)
        if n == 0:
            return [], []

        if n <= 24:
            stride = 1
        elif n <= 60:
            stride = 3
        else:
            stride = 6

        tickvals = list(range(0, n, stride))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)

        ticktext = [x_labels[i] for i in tickvals]
        return tickvals, ticktext

    @staticmethod
    def plot_forecast(actual, fitted, forecast, title="Forecast", x_labels=None):
        """
        Plot actual, fitted, and forecast values with proper connection
        
        Args:
            actual: Historical actual values
            fitted: Fitted values (same length as actual)
            forecast: Future forecast values
        """
        fig = go.Figure()

        # Map x-positions to time labels for hover (falls back to '' out of range)
        def _hover(xs):
            if not x_labels:
                return None, None
            customdata = [x_labels[i] if 0 <= i < len(x_labels) else '' for i in xs]
            return customdata, '%{customdata}<br>%{fullData.name}: %{y:,.2f}<extra></extra>'

        # Actual data
        actual_x = list(range(len(actual)))
        actual_cd, actual_ht = _hover(actual_x)
        fig.add_trace(go.Scatter(
            x=actual_x,
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            customdata=actual_cd,
            hovertemplate=actual_ht
        ))
        
        # Fitted line (if provided)
        if fitted is not None:
            # Fitted line covers the same x-range as actual
            fitted_x = list(range(len(fitted)))
            
            # To connect fitted to forecast, we need to extend the x values
            # The last point of fitted should connect to the first point of forecast
            if forecast is not None:
                # Add the last actual point to the beginning of forecast x-axis
                forecast_start_x = len(actual) - 1
                forecast_x = [forecast_start_x] + list(range(len(actual), len(actual) + len(forecast)))
                
                # Add last fitted value to beginning of forecast for connection
                forecast_y = [fitted[-1]] + list(forecast)

                fitted_cd, fitted_ht = _hover(fitted_x)
                forecast_cd, forecast_ht = _hover(forecast_x)

                # Draw fitted line
                fig.add_trace(go.Scatter(
                    x=fitted_x,
                    y=fitted,
                    mode='lines',
                    name='Fitted',
                    line=dict(color='#A23B72', width=2, dash='dot'),
                    customdata=fitted_cd,
                    hovertemplate=fitted_ht
                ))

                # Draw forecast line (connected to fitted)
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#F18F01', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=forecast_cd,
                    hovertemplate=forecast_ht
                ))
            else:
                # Just fitted, no forecast
                fitted_cd, fitted_ht = _hover(fitted_x)
                fig.add_trace(go.Scatter(
                    x=fitted_x,
                    y=fitted,
                    mode='lines',
                    name='Fitted',
                    line=dict(color='#A23B72', width=2, dash='dot'),
                    customdata=fitted_cd,
                    hovertemplate=fitted_ht
                ))
        else:
            # No fitted line, just forecast
            if forecast is not None:
                forecast_x = list(range(len(actual) - 1, len(actual) + len(forecast) - 1))
                forecast_cd, forecast_ht = _hover(forecast_x)
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#F18F01', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=forecast_cd,
                    hovertemplate=forecast_ht
                ))
        
        xaxis_title = 'Period'
        if x_labels:
            xaxis_title = 'Time'

        # With time labels, use closest-point hover so each tooltip shows the
        # time label (customdata) instead of the numeric x-position header that
        # 'x unified' would display. Without labels, keep the unified readout.
        hovermode = 'closest' if x_labels else 'x unified'

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title='Value',
            hovermode=hovermode,
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        if x_labels:
            tickvals, ticktext = Visualizer._thin_ticks(x_labels)
            fig.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
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