# traffic_analysis_dashboard/vehicle_analyzer.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import gc
import logging
from typing import List, Dict

# Configure logging (consider making this configurable from main)
logger = logging.getLogger(__name__)

class VehicleAnalyzer:
    """
    Optimized vehicle detection and analysis class with memory management
    and performance optimizations.
    """
    
    def __init__(self):
        self.model = None
        self.class_names = {}
        # Define vehicle classes to focus on
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
            raise RuntimeError(f"Failed to load YOLO model: {e}. Please check your internet connection or model path.")
    
    def is_vehicle(self, class_name: str) -> bool:
        """Check if detected object is a vehicle."""
        return class_name.lower() in self.vehicle_classes
    
    def process_frame_batch(self, frames: List[np.ndarray], frame_ids: List[int],
                            confidence_thresh: float) -> List[Dict]:
        """
        Process multiple frames in batch for better performance.
        
        Args:
            frames: List of video frames
            frame_ids: Corresponding frame IDs
            confidence_thresh: Minimum confidence to consider a detection.
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Batch inference for better GPU utilization
            # Added stream=True for potentially faster processing if available
            results = self.model(frames, verbose=False, stream=True) 
            
            for result, frame_id in zip(results, frame_ids):
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        confidence = float(box.conf.cpu().numpy())
                        class_name = self.class_names.get(cls_id, "unknown") # Use .get() for safer access
                        
                        # Only include vehicles with high confidence
                        if self.is_vehicle(class_name) and confidence > confidence_thresh:
                            detections.append({
                                'frame_id': frame_id,
                                'vehicle_type': class_name,
                                'confidence': confidence,
                                'timestamp': frame_id / 30.0  # Assuming 30 FPS for timestamp calculation
                            })
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
            # Optionally re-raise or handle more gracefully
        
        return detections
    
    def analyze_video_optimized(self, video_path: str, 
                                frame_skip: int, 
                                max_frames: int,
                                batch_size: int,
                                confidence_thresh: float,
                                resize_dim: int,
                                fps_hint: float) -> pd.DataFrame:
        """
        Optimized video analysis with batch processing and memory management.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every N-th frame
            max_frames: Maximum frames to process
            batch_size: Number of frames to process in each batch
            confidence_thresh: Minimum confidence for vehicle detection.
            resize_dim: Dimension (width/height) to resize frames for inference.
            fps_hint: Estimated FPS of the video, used for validation and info.
            
        Returns:
            DataFrame with detection results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_to_display = video_fps if video_fps > 0 else fps_hint

        st.info(f"Video info: {total_video_frames} frames, {fps_to_display:.1f} FPS")
        
        all_detections = []
        frame_batch = []
        frame_id_batch = []
        current_frame = 0
        processed_frames_count = 0
        
        # Motion detection for faster skipping of static scenes
        prev_gray_frame = None 
        motion_threshold = 5000 # Tune this value based on video characteristics
        
        # Streamlit elements for progress feedback must be in the main app,
        # so we pass them or use a callback/event model if direct access isn't feasible.
        # For simplicity here, we assume direct access for progress_bar and status_placeholder
        # or mock them if running in isolation. In Streamlit, these will be defined in main.py
        # and passed into this method or handled via global access (less ideal).
        # We will pass them as arguments from main.py.
        # If not passed, create placeholders for testing outside Streamlit.
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()


        while cap.isOpened() and processed_frames_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply motion detection
            if prev_gray_frame is not None:
                frame_diff = cv2.absdiff(current_gray_frame, prev_gray_frame)
                non_zero_pixels = np.count_nonzero(frame_diff)
                
                # Skip this frame if motion is below threshold AND it's not a frame we MUST process due to frame_skip
                if non_zero_pixels < motion_threshold and current_frame % frame_skip != 0:
                    current_frame += 1
                    prev_gray_frame = current_gray_frame
                    del frame
                    continue
            
            prev_gray_frame = current_gray_frame
            
            if current_frame % frame_skip == 0:
                resized_frame = cv2.resize(frame, (resize_dim, resize_dim)) 
                frame_batch.append(resized_frame)
                frame_id_batch.append(current_frame)
                
                if len(frame_batch) >= batch_size:
                    start_process_frame_idx = current_frame - len(frame_batch) * frame_skip # Approximate start frame
                    status_text_placeholder.text(f"Processing frames {start_process_frame_idx} to {current_frame} "
                                                 f"({processed_frames_count}/{max_frames} processed)")
                    
                    batch_detections = self.process_frame_batch(frame_batch, frame_id_batch, confidence_thresh) 
                    all_detections.extend(batch_detections)
                    
                    frame_batch.clear()
                    frame_id_batch.clear()
                    gc.collect()
                    
                    processed_frames_count += batch_size
                    progress_bar_placeholder.progress(min(processed_frames_count / max_frames, 1.0))
            
            current_frame += 1
            del frame
        
        # Process any remaining frames in the last batch
        if frame_batch:
            batch_detections = self.process_frame_batch(frame_batch, frame_id_batch, confidence_thresh) 
            all_detections.extend(batch_detections)
        
        cap.release()
        
        return pd.DataFrame(all_detections) if all_detections else pd.DataFrame()

