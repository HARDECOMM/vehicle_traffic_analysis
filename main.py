import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
import plotly.express as px
import gc
import time
import os # Import os for cleanup

# --- Page Configuration (Optional but Recommended for Dashboards) ---
st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    layout="wide", # Use wide layout to give more space for charts
    initial_sidebar_state="expanded"
)

st.title("Vehicle Traffic Analysis Dashboard")
st.markdown("Upload a video to analyze vehicle types, distribution, and traffic density over time.")

# --- File Uploader ---
uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_video:
    if st.button("Start Analysis"):
        with st.spinner("Processing video and generating insights... This may take a moment."):
            # --- 1. Save uploaded file to temporary storage ---
            # Using tempfile.NamedTemporaryFile for safer handling
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_file.write(uploaded_video.read())
            video_path = temp_video_file.name
            temp_video_file.close() # Close the file immediately after writing

            # --- 2. Load YOLO model ---
            try:
                # Cache the model to avoid reloading on every rerun
                @st.cache_resource
                def load_yolo_model():
                    return YOLO('yolov8n.pt') # yolov8n is lightweight and faster
                model = load_yolo_model()
            except Exception as e:
                st.error(f"Error loading YOLO model: {e}")
                st.stop() # Stop execution if model fails to load

            # --- 3. Process video frames and collect data ---
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Error: Could not open video at {video_path}. Please check file integrity.")
                # Clean up temp file
                os.unlink(video_path)
                st.stop()

            frame_skip = st.slider("Process every N-th frame (higher = faster, lower = more detailed)", 1, 10, 5) # Slider for user control
            max_frames_to_process = st.number_input("Maximum frames to analyze (for quick demo)", 10, 500, 100) # User can set max frames


            # Data structures to store results
            all_detections_data = [] # Stores {'frame', 'class', 'confidence'} for each detection
            class_names = model.names # Get names from YOLO model for plotting

            current_frame_id = 0
            processed_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: # End of video
                    break

                if current_frame_id % frame_skip == 0:
                    status_text.text(f"Analyzing frame {current_frame_id}...")
                    # Resize frame for faster inference (can be adjusted)
                    # Common sizes: 320x320, 480x480, 640x640
                    frame_for_inference = cv2.resize(frame, (320, 320))

                    # Perform inference
                    results = model(frame_for_inference, verbose=False)
                    detections = results[0] # Get detections for the current frame

                    # Process detected bounding boxes
                    if detections.boxes is not None:
                        for box in detections.boxes:
                            cls_id = int(box.cls.cpu().numpy())
                            confidence = float(box.conf.cpu().numpy())
                            # Store detection data
                            all_detections_data.append({
                                'frame_id': current_frame_id,
                                'vehicle_type': class_names[cls_id],
                                'confidence': confidence
                            })
                    processed_count += 1
                    progress_bar.progress(min(processed_count / max_frames_to_process, 1.0))

                    if processed_count >= max_frames_to_process:
                        status_text.text(f"Reached maximum frames ({max_frames_to_process}). Stopping analysis.")
                        break # Stop if max frames reached

                current_frame_id += 1
                # Clean up memory
                del frame
                gc.collect()

            cap.release()
            os.unlink(video_path) # Clean up temporary video file

            # --- 4. Generate Dashboard Visualizations ---
            st.success("Analysis Complete! Generating Dashboard.")

            if not all_detections_data:
                st.warning("No vehicles were detected in the processed video. Try a different video or adjust settings.")
                st.stop()

            df_detections = pd.DataFrame(all_detections_data)

            # --- KPI: Total Vehicles Detected (unique detections) ---
            st.header("Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(df_detections))
            with col2:
                # Assuming 'vehicle_type' represents distinct categories
                st.metric("Unique Vehicle Types", df_detections['vehicle_type'].nunique())
            with col3:
                # Simple average density per processed frame
                avg_density = len(df_detections) / processed_count if processed_count > 0 else 0
                st.metric(f"Avg. Vehicles per {frame_skip}-th Frame", f"{avg_density:.2f}")

            # --- Chart 1: Vehicle Type Distribution (Bar Chart) ---
            st.header("Vehicle Type Distribution")
            vehicle_counts = df_detections['vehicle_type'].value_counts().reset_index()
            vehicle_counts.columns = ['Vehicle Type', 'Count']
            fig_type_bar = px.bar(
                vehicle_counts,
                x='Vehicle Type',
                y='Count',
                title='Count of Vehicles by Type',
                text='Count',
                template='plotly_white'
            )
            st.plotly_chart(fig_type_bar, use_container_width=True)

            # --- Chart 2: Proportion of Vehicle Types (Pie Chart) ---
            st.header("Proportion of Vehicle Types")
            fig_type_pie = px.pie(
                vehicle_counts,
                names='Vehicle Type',
                values='Count',
                title='Proportion of Each Vehicle Type',
                template='plotly_white'
            )
            st.plotly_chart(fig_type_pie, use_container_width=True)

            # --- Chart 3: Traffic Density Over Time (Line Chart) ---
            st.header("Traffic Density Over Time")
            # Group by frame_id to get count of vehicles in each processed frame
            traffic_over_time = df_detections.groupby('frame_id').size().reset_index(name='vehicles_detected')
            fig_time = px.line(
                traffic_over_time,
                x='frame_id',
                y='vehicles_detected',
                title=f'Number of Vehicles Detected per {frame_skip}-th Frame',
                labels={'frame_id': 'Frame Number', 'vehicles_detected': 'Vehicles Detected'},
                markers=True,
                template='plotly_white'
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # --- Download Raw Data ---
            st.header("Download Data")
            st.download_button(
                label="Download All Detections Data (CSV)",
                data=df_detections.to_csv(index=False).encode('utf-8'),
                file_name='all_vehicle_detections.csv',
                mime='text/csv'
            )
            st.download_button(
                label="Download Vehicle Type Counts (CSV)",
                data=vehicle_counts.to_csv(index=False).encode('utf-8'),
                file_name='vehicle_type_counts.csv',
                mime='text/csv'
            )

            st.info("Analysis complete! Scroll up to see the dashboard.")