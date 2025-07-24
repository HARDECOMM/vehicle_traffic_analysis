# traffic_analysis_dashboard/main.py

import streamlit as st
import cv2
import pandas as pd
import tempfile
import gc
import time
import os
import logging

from vehicle_analyzer import VehicleAnalyzer
from dashboard_generator import DashboardGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Page configuration
    st.set_page_config(
        page_title="🚗 Traffic Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Vehicle Traffic Analysis Dashboard")
    st.markdown("Upload a video to analyze vehicle types, distribution, and traffic patterns in seconds!")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        frame_skip = st.slider("Frame Skip (Higher = Faster)", 1, 15, 5) 
        max_frames = st.slider("Max Frames to Analyze", 50, 2000, 500) 
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1) 
        confidence_thresh = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.3, 0.05) 
        resize_dim = st.selectbox("Frame Resize Dimension (px)", [320, 416, 640], index=2, 
                                  help="Smaller = faster processing, less accuracy.")
        
        st.header("📊 Performance Tips")
        st.info("""
        - Higher frame skip speeds up but risks missing vehicles.
        - Lower max frames makes results quicker.
        - Larger batch size better uses GPU but needs more memory.
        - Lower confidence threshold finds more objects, higher reduces false positives.
        - Smaller resize speeds up detection but may reduce accuracy.
        """)

    # --- Session-state enhanced file uploader ---
    uploaded_video_new = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
        key="video_uploader"
    )
    if uploaded_video_new is not None:
        st.session_state['uploaded_video'] = uploaded_video_new

    uploaded_video = st.session_state.get('uploaded_video', None)

    total_video_frames = 0
    video_fps = 30.0  # default fallback

    if uploaded_video:
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)
        st.info(f"📁 File: {uploaded_video.name} ({file_size:.1f} MB)")

        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_video_path = temp_file.name
            
            cap_info = cv2.VideoCapture(temp_video_path)
            total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap_info.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0  # fallback
            cap_info.release()

            if frame_skip >= video_fps:
                st.warning(f"⚠️ Frame Skip ({frame_skip}) >= video FPS ({video_fps:.1f}). Consider reducing frame skip to avoid missing detections.")

            est_processed_frames = min(max_frames, total_video_frames) // frame_skip
            if est_processed_frames < 50:
                st.warning(f"⚠️ Estimated frames to process ({est_processed_frames}) is low, consider increasing max frames or reducing frame skip.")

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
                        fps_hint=video_fps
                    )
                analysis_time = time.time() - start_time

                if df_results.empty:
                    st.warning("⚠️ No vehicles detected. Try adjusting settings or use a different video.")
                else:
                    st.success(f"✅ Analysis complete in {analysis_time:.1f} seconds!")

                    dashboard = DashboardGenerator()
                    st.header("📈 Key Metrics")
                    dashboard.create_kpi_metrics(df_results, est_processed_frames, frame_skip)

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
