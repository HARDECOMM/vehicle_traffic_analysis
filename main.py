# main.py

import streamlit as st
import pandas as pd
import gc
import logging

from vehicle_analyzer import VehicleAnalyzer
import dashboard_generator as dg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="🚗Vehicle Traffic Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚗Vehicle Traffic Analysis Dashboard")
    st.markdown("Upload a video to analyze vehicle types, distribution, and traffic density over time.")
    
    # Sidebar inputs with concise descriptions
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        frame_skip = st.slider(
            "Process every N-th frame",
            min_value=1,
            max_value=15,
            value=5,
            help="Higher values skip more frames and speed up analysis but may miss detections. Lower values analyze more frames for detailed results."
        )
        st.caption("Frame skipping trades off between speed and detail.")
        
        max_frames = st.number_input(
            "Maximum frames to analyze",
            min_value=10,
            max_value=2000,
            value=500,
            step=10,
            help="Sets the upper limit on processed video frames to control runtime and speed."
        )
        st.caption("Limit frames to reduce processing time on longer videos.")
    
    # File uploader with session state persistence
    uploaded_video_new = st.file_uploader(
        "Upload a traffic video", 
        type=["mp4", "avi", "mov"], 
        key="video_uploader"
    )
    if uploaded_video_new is not None:
        st.session_state['uploaded_video'] = uploaded_video_new
    uploaded_video = st.session_state.get('uploaded_video', None)
    
    if uploaded_video:
        st.info(f"Uploaded file: {uploaded_video.name} ({uploaded_video.size / (1024*1024):.2f} MB)")
        analyzer = VehicleAnalyzer()
        
        if st.button("Start Analysis"):
            video_path = None
            try:
                video_path = analyzer.save_uploaded_video(uploaded_video)
                analyzer.load_model()
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_cb(progress):
                    progress_bar.progress(min(progress, 1.0))
                
                detections = analyzer.analyze_video(video_path, frame_skip, max_frames, progress_cb)
                status_text.text("Analysis complete.")
                
                if not detections:
                    st.warning("No vehicles detected. Try a different video or adjust parameters.")
                    return
                
                df = pd.DataFrame(detections)

                dg.show_key_metrics(df, max_frames, frame_skip)
                dg.plot_vehicle_type_distribution(df)
                dg.plot_vehicle_type_pie(df)
                dg.plot_traffic_density_over_time(df, frame_skip)

                st.header("Download Data")
                st.download_button(
                    "Download All Detections (CSV)",
                    df.to_csv(index=False).encode('utf-8'),
                    "detections.csv",
                    "text/csv"
                )
                vehicle_counts = df['vehicle_type'].value_counts().reset_index()
                vehicle_counts.columns = ['Vehicle Type', 'Count']
                st.download_button(
                    "Download Vehicle Type Summary (CSV)",
                    vehicle_counts.to_csv(index=False).encode('utf-8'),
                    "vehicle_summary.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logger.error(f"Analysis error: {e}")
            finally:
                if video_path:
                    analyzer.cleanup_temp_file(video_path)
                gc.collect()

if __name__ == "__main__":
    main()
