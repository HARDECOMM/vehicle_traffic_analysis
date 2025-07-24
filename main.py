"""
Optimized Vehicle Traffic Analysis Dashboard

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
            # Using yolov8n.pt for efficiency
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
    
    def process_frame_batch(self, frames: List[np.ndarray], frame_ids: List[int],
                            confidence_thresh: float = 0.3) -> List[Dict]: # MOD: Added confidence_thresh parameter
        """
        Process multiple frames in batch for better performance.
        
        Args:
            frames: List of video frames
            frame_ids: Corresponding frame IDs
            confidence_thresh: Minimum confidence to consider a detection. # MOD: Added param description
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Batch inference for better GPU utilization
            # Added stream=True for potentially faster processing if available
            results = self.model(frames, verbose=False, stream=True) 
            
            for i, (result, frame_id) in enumerate(zip(results, frame_ids)):
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        confidence = float(box.conf.cpu().numpy())
                        class_name = self.class_names[cls_id]
                        
                        # Only include vehicles with high confidence
                        if self.is_vehicle(class_name) and confidence > confidence_thresh: # MOD: Use passed confidence_thresh
                            detections.append({
                                'frame_id': frame_id,
                                'vehicle_type': class_name,
                                'confidence': confidence,
                                'timestamp': frame_id / 30.0  # Assuming 30 FPS for timestamp calculation
                            })
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
        
        return detections
    
    def analyze_video_optimized(self, video_path: str, 
                              frame_skip: int = 5, 
                              max_frames: int = 300,
                              batch_size: int = 8,
                              confidence_thresh: float = 0.3, # MOD: Added confidence_thresh param
                              resize_dim: int = 640, # MOD: Added resize_dim param
                              fps_hint: float = 30.0) -> pd.DataFrame: # MOD: Added fps_hint for validation
        """
        Optimized video analysis with batch processing and memory management.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every N-th frame
            max_frames: Maximum frames to process
            batch_size: Number of frames to process in each batch
            confidence_thresh: Minimum confidence for vehicle detection. # MOD: Added param description
            resize_dim: Dimension (width/height) to resize frames for inference. # MOD: Added param description
            fps_hint: Estimated FPS of the video, used for validation and info. # MOD: Added param description
            
        Returns:
            DataFrame with detection results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties for optimization
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) # Actual FPS from video
        fps_to_display = video_fps if video_fps > 0 else fps_hint # Use actual FPS if available

        st.info(f"Video info: {total_frames} frames, {fps_to_display:.1f} FPS")
        
        all_detections = []
        frame_batch = []
        frame_id_batch = []
        current_frame = 0
        processed_frames_count = 0 # MOD: Renamed for clarity to avoid confusion with current_frame
        
        # MOD: Add previous frame for motion detection
        prev_gray_frame = None 
        motion_threshold = 5000 # Tune this value based on video characteristics
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        while cap.isOpened() and processed_frames_count < max_frames: # MOD: Use processed_frames_count
            ret, frame = cap.read()
            if not ret:
                break
            
            # MOD: Motion detection logic
            current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray_frame is not None:
                frame_diff = cv2.absdiff(current_gray_frame, prev_gray_frame)
                non_zero_pixels = np.count_nonzero(frame_diff)
                
                # If motion is below threshold and we are not forced to process by frame_skip
                if non_zero_pixels < motion_threshold and current_frame % frame_skip != 0:
                    current_frame += 1
                    prev_gray_frame = current_gray_frame
                    del frame # Clean up
                    continue # Skip this frame, not enough motion
            
            prev_gray_frame = current_gray_frame # Update for next iteration
            
            if current_frame % frame_skip == 0:
                # Resize for faster inference
                # MOD: Use resize_dim from parameter
                resized_frame = cv2.resize(frame, (resize_dim, resize_dim)) 
                frame_batch.append(resized_frame)
                frame_id_batch.append(current_frame)
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    # MOD: Improved status message
                    status_placeholder.text(f"Processing frames {current_frame - len(frame_batch)*frame_skip} to {current_frame} "
                                            f"({processed_frames_count}/{max_frames} processed)")
                    
                    # MOD: Pass confidence_thresh to process_frame_batch
                    batch_detections = self.process_frame_batch(frame_batch, frame_id_batch, confidence_thresh) 
                    all_detections.extend(batch_detections)
                    
                    # Clear batch and free memory
                    frame_batch.clear()
                    frame_id_batch.clear()
                    gc.collect()
                    
                    processed_frames_count += batch_size # MOD: Use processed_frames_count
                    progress_bar.progress(min(processed_frames_count / max_frames, 1.0)) # MOD: Use processed_frames_count
            
            current_frame += 1
            del frame  # Explicit memory cleanup
        
        # Process remaining frames in batch
        if frame_batch:
            # MOD: Pass confidence_thresh to process_frame_batch
            batch_detections = self.process_frame_batch(frame_batch, frame_id_batch, confidence_thresh) 
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
            fig_pie.update_traces(textposition='inside', textinfo='percent+label') # Added text info for pie chart
            st.plotly_chart(fig_pie, use_container_width=True)
    
    @staticmethod
    def create_traffic_timeline(df: pd.DataFrame) -> None:
        """Create traffic density timeline visualization."""
        if df.empty:
            st.warning("No data available for timeline chart.")
            return
        
        # Traffic density over time
        # Ensure timestamp is calculated based on video's actual FPS if possible, or 30 FPS
        # Passed fps_hint can be used if video_fps is not reliable
        traffic_timeline = df.groupby('frame_id').size().reset_index(name='vehicle_count')
        # Assuming original 30 FPS, but if actual FPS is known and passed down, use it for more accurate time
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
        # MOD: Default frame_skip to 5 for slightly more frequent analysis
        frame_skip = st.slider("Frame Skip (Higher = Faster)", 1, 15, 5) 
        max_frames = st.slider("Max Frames to Analyze", 50, 2000, 500) # MOD: Increased max_frames range and default
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1) # MOD: Added 32 to batch size options
        # MOD: Added confidence threshold slider with lower default
        confidence_thresh = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.3, 0.05) 
        # MOD: Added resize dimension selection
        resize_dim = st.selectbox("Frame Resize Dimension (px)", [320, 416, 640], index=2, 
                                  help="Smaller dimensions process faster but may lose accuracy.") 
        
        st.header("📊 Performance Tips")
        st.info("""
        - **Frame Skip:** Higher skips frames, speeding up analysis but potentially missing vehicles.
        - **Max Frames:** Limits total frames processed. Lower for quicker tests.
        - **Batch Size:** Larger uses GPU more efficiently but requires more memory.
        - **Confidence Threshold:** Lower detects more objects (even weak ones), higher reduces false positives.
        - **Resize Dimension:** Smaller frame size means faster processing but can reduce detection accuracy.
        - *Motion detection is applied to skip frames with no activity.*
        """)
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    # Initialize fps_for_validation
    fps_for_validation = 30.0 

    if uploaded_video:
        # Display video info
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 File: {uploaded_video.name} ({file_size:.1f} MB)")
        
        # MOD: Pre-read video info to validate parameters
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_video_path = temp_file.name
            
            cap_info = cv2.VideoCapture(temp_video_path)
            total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap_info.get(cv2.CAP_PROP_FPS)
            fps_for_validation = video_fps if video_fps > 0 else 30.0 # Use actual FPS for validation
            cap_info.release()

            # MOD: Parameter validation based on video properties
            if frame_skip >= fps_for_validation:
                st.warning(f"⚠️ **Warning:** Frame Skip ({frame_skip}) is very high compared to video FPS ({fps_for_validation:.1f}). "
                           "This may lead to significant loss of information or 'No vehicles detected'. Consider lowering it.")
            
            estimated_processed_frames = min(max_frames, total_video_frames // frame_skip)
            if estimated_processed_frames < 50: # MOD: Warn if too few frames will be processed
                 st.warning(f"⚠️ **Warning:** With current settings, only ~{estimated_processed_frames} frames will be processed. "
                            "This might result in 'No vehicles detected'. Consider increasing 'Max Frames to Analyze' or lowering 'Frame Skip'.")


        except Exception as e:
            st.error(f"Error reading video info: {e}")
            logger.error(f"Error reading video info: {e}")
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            return # Stop processing if video info cannot be read
        
        if st.button("🚀 Start Analysis", type="primary"):
            start_time = time.time()
            
            # Initialize analyzer
            try:
                analyzer = VehicleAnalyzer()
                analyzer.model = analyzer.load_model()
            except Exception as e:
                st.error(f"Failed to initialize analyzer: {e}")
                st.stop()
            
            # Ensure video_path is set from temp_video_path
            video_path = temp_video_path # Use the path from validation step
            
            try:
                with st.spinner("🔍 Analyzing video... This will take just a few seconds!"):
                    # Perform analysis
                    # MOD: Pass confidence_thresh, resize_dim, and fps_for_validation to analyze_video_optimized
                    df_results = analyzer.analyze_video_optimized(
                        video_path, frame_skip, max_frames, batch_size,
                        confidence_thresh=confidence_thresh, resize_dim=resize_dim,
                        fps_hint=fps_for_validation # Pass actual video FPS or default
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
                    # MOD: Pass processed_frames_count for KPI (using max_frames if actual processed is less)
                    processed_frames_for_kpi = min(max_frames, total_video_frames) // frame_skip # Approximate count for KPI
                    dashboard.create_kpi_metrics(df_results, processed_frames_for_kpi, frame_skip)
                    
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
                        # Use .copy() to avoid SettingWithCopyWarning if df_results is a slice
                        summary_data = df_results.groupby('vehicle_type').agg(
                            count=('confidence', 'count'), # Named aggregation
                            mean_confidence=('confidence', 'mean'), # Named aggregation
                            min_frame_id=('frame_id', 'min'), # Named aggregation
                            max_frame_id=('frame_id', 'max') # Named aggregation
                        ).round(3)
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

