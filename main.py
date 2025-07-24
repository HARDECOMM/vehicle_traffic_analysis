# main.py

import streamlit as st
import pandas as pd
import gc
import os
import logging

from vehicle_analyzer import VehicleAnalyzer
import dashboard_generator as dg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="Traffic Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Vehicle Traffic Analysis Dashboard")
    st.markdown("Upload a video to analyze vehicle types, distribution, and traffic density over time.")
    
    # Sidebar controls
    with st.sidebar:
        frame_skip = st.slider("Process every N-th frame", 1, 10, 5)
        max_frames = st.number_input("Max frames to analyze", min_value=10, max_value=500, value=100)
    
    # File uploader with session state to persist upload
    uploaded_video_new = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded_video_new is not None:
        st.session_state['uploaded_video'] = uploaded_video_new
    
    uploaded_video = st.session_state.get('uploaded_video', None)
    
    if uploaded_video is not None:
        st.info(f"Uploaded file: {uploaded_video.name} ({uploaded_video.size / (1024*1024):.2f} MB)")
        
        analyzer = VehicleAnalyzer()
        model = None
        video_path = None
        
        if st.button("Start Analysis"):
            # Save video to temp file
            video_path = analyzer.save_uploaded_video(uploaded_video)
            
            # Load model
            try:
                model = analyzer.load_model()
            except Exception as e:
                st.error(f"Failed to load YOLO model: {e}")
                return
            
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback for analyzer
            def progress_callback(progress):
                progress_bar.progress(min(progress, 1.0))
            
            # Run analysis
            try:
                detections = analyzer.analyze_video(video_path, frame_skip, max_frames, progress_callback)
                status_text.text("Analysis complete.")
                if not detections:
                    st.warning("No vehicles detected. Try a different video or adjust parameters.")
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(detections)
                
                # Show KPIs and charts
                dg.show_key_metrics(df, max_frames, frame_skip)
                dg.plot_vehicle_type_distribution(df)
                dg.plot_vehicle_type_pie(df)
                dg.plot_traffic_density_over_time(df, frame_skip)
                
                # Download options
                st.header("Download Data")
                csv_all = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download All Detections (CSV)", csv_all, "detections.csv", "text/csv")
                
                # Vehicle counts summary csv file
                vehicle_counts = df['vehicle_type'].value_counts().reset_index()
                vehicle_counts.columns = ['Vehicle Type', 'Count']
                csv_summary = vehicle_counts.to_csv(index=False).encode('utf-8')
                st.download_button("Download Vehicle Type Summary (CSV)", csv_summary, "vehicle_summary.csv", "text/csv")
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logger.error(f"Analysis error: {e}")
            
            finally:
                # Cleanup temp video file
                if video_path:
                    analyzer.cleanup_temp_file(video_path)
                gc.collect()

if __name__ == "__main__":
    main()
