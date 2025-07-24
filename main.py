"""
This module provides an efficient Streamlit interface for real-time vehicle detection
and analysis using YOLO models with memory optimization and error handling.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import gc
import time
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleAnalyzer:
    """
    Optimized vehicle detection and analysis class with memory management
    and performance optimizations.
    """
    
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
        
    @st.cache_resource
    def load_model(_self):
        """Load YOLO model with caching for deployment efficiency."""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            _self.class_names = model.names
            logger.info("YOLO model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def is_vehicle(self, class_name: str) -> bool:
        """Check if detected object is a vehicle."""
        return class_name.lower() in self.vehicle_classes
    
    def process_frame_batch(self, frames: List[np.ndarray], frame_ids: List[int]) -> List[Dict]:
        """
        Process multiple frames in batch for better performance.
        
        Args:
            frames: List of video frames
            frame_ids: Corresponding frame IDs
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Batch inference for better GPU utilization
            results = self.model(frames, verbose=False)
            
            for i, (result, frame_id) in enumerate(zip(results, frame_ids)):
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        confidence = float(box.conf.cpu().numpy())
                        class_name = self.class_names[cls_id]
                        
                        # Only include vehicles with high confidence
                        if self.is_vehicle(class_name) and confidence > 0.5:
                            detections.append({
                                'frame_id': frame_id,
                                'vehicle_type': class_name,
                                'confidence': confidence,
                                'timestamp': frame_id / 30.0  # Assuming 30 FPS
                            })
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
        
        return detections
    
    def analyze_video_optimized(self, video_path: str, 
                              frame_skip: int = 5, 
                              max_frames: int = 300,
                              batch_size: int = 8) -> pd.DataFrame:
        """
        Optimized video analysis with batch processing and memory management.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every N-th frame
            max_frames: Maximum frames to process
            batch_size: Number of frames to process in each batch
            
        Returns:
            DataFrame with detection results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties for optimization
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.info(f"Video info: {total_frames} frames, {fps:.1f} FPS")
        
        all_detections = []
        frame_batch = []
        frame_id_batch = []
        current_frame = 0
        processed_frames = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_skip == 0:
                # Resize for faster inference
                resized_frame = cv2.resize(frame, (640, 640))
                frame_batch.append(resized_frame)
                frame_id_batch.append(current_frame)
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    status_placeholder.text(f"Processing frames {current_frame-batch_size*frame_skip} to {current_frame}")
                    
                    batch_detections = self.process_frame_batch(frame_batch, frame_id_batch)
                    all_detections.extend(batch_detections)
                    
                    # Clear batch and free memory
                    frame_batch.clear()
                    frame_id_batch.clear()
                    gc.collect()
                    
                    processed_frames += batch_size
                    progress_bar.progress(min(processed_frames / max_frames, 1.0))
            
            current_frame += 1
            del frame  # Explicit memory cleanup
        
        # Process remaining frames in batch
        if frame_batch:
            batch_detections = self.process_frame_batch(frame_batch, frame_id_batch)
            all_detections.extend(batch_detections)
        
        cap.release()
        
        return pd.DataFrame(all_detections) if all_detections else pd.DataFrame()

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
            st.plotly_chart(fig_pie, use_container_width=True)
    
    @staticmethod
    def create_traffic_timeline(df: pd.DataFrame) -> None:
        """Create traffic density timeline visualization."""
        if df.empty:
            st.warning("No data available for timeline chart.")
            return
        
        # Traffic density over time
        traffic_timeline = df.groupby('frame_id').size().reset_index(name='vehicle_count')
        traffic_timeline['timestamp'] = traffic_timeline['frame_id'] / 30.0  # Convert to seconds
        
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
            template='plotly_white'
        )
        fig.update_layout(xaxis_title='Confidence Score', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function with optimized structure."""
    
    # Page configuration
    st.set_page_config(
        page_title="🚗 Traffic Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Vehicle Traffic Analysis Dashboard")
    st.markdown("Upload a video to analyze vehicle types, distribution, and traffic patterns in seconds!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        frame_skip = st.slider("Frame Skip (Higher = Faster)", 1, 15, 8)
        max_frames = st.slider("Max Frames to Analyze", 50, 1000, 300)
        batch_size = st.selectbox("Batch Size", [4, 8, 16], index=1)
        
        st.header("📊 Performance Tips")
        st.info("""
        - Higher frame skip = faster analysis
        - Lower max frames = quicker results
        - Larger batch size = better GPU utilization
        """)
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video:
        # Display video info
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 File: {uploaded_video.name} ({file_size:.1f} MB)")
        
        if st.button("🚀 Start Analysis", type="primary"):
            start_time = time.time()
            
            # Initialize analyzer
            try:
                analyzer = VehicleAnalyzer()
                analyzer.model = analyzer.load_model()
            except Exception as e:
                st.error(f"Failed to initialize analyzer: {e}")
                st.stop()
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.getvalue())
                video_path = temp_file.name
            
            try:
                with st.spinner("🔍 Analyzing video... This will take just a few seconds!"):
                    # Perform analysis
                    df_results = analyzer.analyze_video_optimized(
                        video_path, frame_skip, max_frames, batch_size
                    )
                
                analysis_time = time.time() - start_time
                
                if df_results.empty:
                    st.warning("⚠️ No vehicles detected. Try adjusting settings or use a different video.")
                else:
                    st.success(f"✅ Analysis complete in {analysis_time:.1f} seconds!")
                    
                    # Generate dashboard
                    dashboard = DashboardGenerator()
                    
                    # KPI Section
                    st.header("📈 Key Metrics")
                    dashboard.create_kpi_metrics(df_results, max_frames // frame_skip, frame_skip)
                    
                    # Visualizations
                    st.header("📊 Vehicle Distribution")
                    dashboard.create_vehicle_distribution_chart(df_results)
                    
                    st.header("📈 Traffic Timeline")
                    dashboard.create_traffic_timeline(df_results)
                    
                    st.header("🎯 Detection Quality")
                    dashboard.create_confidence_analysis(df_results)
                    
                    # Data export
                    st.header("💾 Export Data")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Data (CSV)",
                            csv_data,
                            f"traffic_analysis_{int(time.time())}.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        summary_data = df_results.groupby('vehicle_type').agg({
                            'confidence': ['count', 'mean'],
                            'frame_id': ['min', 'max']
                        }).round(3)
                        st.download_button(
                            "📊 Download Summary (CSV)",
                            summary_data.to_csv(),
                            f"traffic_summary_{int(time.time())}.csv",
                            "text/csv"
                        )
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Analysis error: {e}")
            
            finally:
                # Cleanup
                if os.path.exists(video_path):
                    os.unlink(video_path)
                gc.collect()

if __name__ == "__main__":
    main()