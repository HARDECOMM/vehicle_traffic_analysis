# traffic_analysis_dashboard/main.py

import streamlit as st
import cv2
import pandas as pd
import tempfile
import gc
import time
import os
import logging

# Import your custom modules
from vehicle_analyzer import VehicleAnalyzer
from dashboard_generator import DashboardGenerator

# Configure logging (good place for overall app logging configuration)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        frame_skip = st.slider("Frame Skip (Higher = Faster)", 1, 15, 5) 
        max_frames = st.slider("Max Frames to Analyze", 50, 2000, 500) 
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1) 
        confidence_thresh = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.3, 0.05) 
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
    
    # --- Modified File Uploader with Session State ---
    
    uploaded_video_new = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
        key="video_uploader"
    )
    
    if uploaded_video_new is not None:
        st.session_state['uploaded_video'] = uploaded_video_new
    
    uploaded_video = st.session_state.get('uploaded_video', None)
    
    # Initialize variables for video info
    total_video_frames = 0
    video_fps = 30.0 # Default if not read

    if uploaded_video:
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)
        st.info(f"📁 File: {uploaded_video.name} ({file_size:.1f} MB)")
        
        temp_video_path = None
        try:
            # Save uploaded file to a temporary location for OpenCV to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_video_path = temp_file.name
            
            cap_info = cv2.VideoCapture(temp_video_path)
            total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap_info.get(cv2.CAP_PROP_FPS)
            # Ensure video_fps is a positive number, default to 30 if not available
            video_fps = video_fps if video_fps > 0 else 30.0 
            cap_info.release()

            # Parameter validation warnings
            if frame_skip >= video_fps:
                st.warning(f"⚠️ **Warning:** Frame Skip ({frame_skip}) is very high compared to video FPS ({video_fps:.1f}). "
                           "This may lead to significant loss of information or 'No vehicles detected'. Consider lowering it.")
            
            # Calculate estimated processed frames considering both max_frames and actual video length
            estimated_processed_frames = min(max_frames, total_video_frames) // frame_skip
            if estimated_processed_frames < 50:
                 st.warning(f"⚠️ **Warning:** With current settings, only ~{estimated_processed_frames} frames will be processed. "
                            "This might result in 'No vehicles detected'. Consider increasing 'Max Frames to Analyze' or lowering 'Frame Skip'.")

        except Exception as e:
            st.error(f"Error reading video info: {e}")
            logger.error(f"Error reading video info: {e}")
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            return
        
        if st.button("🚀 Start Analysis", type="primary"):
            start_time = time.time()
            
            try:
                analyzer = VehicleAnalyzer()
                analyzer.model = analyzer.load_model()
            except Exception as e:
                st.error(f"Failed to initialize analyzer: {e}")
                st.stop()
            
            # Use the already saved temp_video_path
            video_path_for_analysis = temp_video_path 
            
            try:
                with st.spinner("🔍 Analyzing video... This will take just a few seconds!"):
                    df_results = analyzer.analyze_video_optimized(
                        video_path_for_analysis, 
                        frame_skip, 
                        max_frames, 
                        batch_size,
                        confidence_thresh=confidence_thresh, 
                        resize_dim=resize_dim,
                        fps_hint=video_fps # Pass the actual video FPS
                    )
                
                analysis_time = time.time() - start_time
                
                if df_results.empty:
                    st.warning("⚠️ No vehicles detected. Try adjusting settings or use a different video.")
                else:
                    st.success(f"✅ Analysis complete in {analysis_time:.1f} seconds!")
                    
                    dashboard = DashboardGenerator()
                    
                    st.header("📈 Key Metrics")
                    # Pass the estimated number of frames that would have been processed for KPI calculations
                    dashboard.create_kpi_metrics(df_results, estimated_processed_frames, frame_skip)
                    
                    st.header("📊 Vehicle Distribution")
                    dashboard.create_vehicle_distribution_chart(df_results)
                    
                    st.header("📈 Traffic Timeline")
                    dashboard.create_traffic_timeline(df_results)
                    
                    st.header("🎯 Detection Quality")
                    dashboard.create_confidence_analysis(df_results)
                    
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
                        summary_data = df_results.groupby('vehicle_type').agg(
                            count=('confidence', 'count'), 
                            mean_confidence=('confidence', 'mean'), 
                            min_frame_id=('frame_id', 'min'), 
                            max_frame_id=('frame_id', 'max') 
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
                if video_path_for_analysis and os.path.exists(video_path_for_analysis):
                    os.unlink(video_path_for_analysis)
                gc.collect()

if __name__ == "__main__":
    main()
