# traffic_analysis_dashboard/dashboard_generator.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class DashboardGenerator:
    """Generate interactive dashboard visualizations with performance optimization."""
    
    @staticmethod
    def create_kpi_metrics(df: pd.DataFrame, processed_frames: int, frame_skip: int) -> None:
        """Create and display KPI metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehicles", len(df))
        with col2:
            st.metric("Vehicle Types", df['vehicle_type'].nunique() if not df.empty else 0)
        with col3:
            # Avoid division by zero if no frames were processed
            avg_per_frame = len(df) / processed_frames if processed_frames > 0 else 0
            st.metric("Avg per Frame", f"{avg_per_frame:.2f}")
        with col4:
            peak_frame = df.groupby('frame_id').size().max() if not df.empty else 0
            st.metric("Peak Traffic", int(peak_frame))
    
    @staticmethod
    def create_vehicle_distribution_chart(df: pd.DataFrame) -> None:
        """Create vehicle type distribution visualization."""
        if df.empty:
            st.warning("No data available for vehicle distribution chart.")
            return
        
        vehicle_counts = df['vehicle_type'].value_counts().reset_index()
        vehicle_counts.columns = ['Vehicle Type', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                vehicle_counts,
                x='Vehicle Type',
                y='Count',
                title='Vehicle Count by Type',
                text='Count',
                color='Vehicle Type',
                template='plotly_white'
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                vehicle_counts,
                names='Vehicle Type',
                values='Count',
                title='Vehicle Type Distribution',
                template='plotly_white'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    @staticmethod
    def create_traffic_timeline(df: pd.DataFrame) -> None:
        """Create traffic density timeline visualization."""
        if df.empty:
            st.warning("No data available for timeline chart.")
            return
        
        # Traffic density over time
        traffic_timeline = df.groupby('frame_id').size().reset_index(name='vehicle_count')
        # Assuming original 30 FPS for timestamp calculation
        traffic_timeline['timestamp'] = traffic_timeline['frame_id'] / 30.0 
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=traffic_timeline['timestamp'],
            y=traffic_timeline['vehicle_count'],
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Traffic Density Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Number of Vehicles',
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_confidence_analysis(df: pd.DataFrame) -> None:
        """Create confidence score analysis."""
        if df.empty:
            return
        
        fig = px.histogram(
            df,
            x='confidence',
            color='vehicle_type',
            title='Detection Confidence Distribution',
            nbins=20,
            template='plotly_white',
            marginal="box" # Added box plot for marginal distribution
        )
        fig.update_layout(xaxis_title='Confidence Score', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

