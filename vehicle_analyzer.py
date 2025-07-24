# vehicle_analyzer.py

import cv2
import gc
import os
import tempfile
from ultralytics import YOLO

class VehicleAnalyzer:
    def __init__(self):
        self.model = None
        self.class_names = []

    def load_model(self):
        # Cache can be applied by the caller if needed
        if self.model is None:
            self.model = YOLO('yolov8n.pt')
            self.class_names = self.model.names
        return self.model

    def analyze_video(self, video_path, frame_skip=5, max_frames=100, progress_callback=None):
        """
        Analyze video, detect vehicles using YOLO model.

        Args:
            video_path (str): Path to video file.
            frame_skip (int): Process every Nth frame.
            max_frames (int): Maximum number of frames to analyze.
            progress_callback (function): Optional, receives progress: float (0-1).

        Returns:
            List of dicts with detection results: {'frame_id', 'vehicle_type', 'confidence'}.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        all_detections = []
        frame_id = 0
        processed_frames = 0

        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_skip == 0:
                # Resize for model input
                resized = cv2.resize(frame, (320, 320))

                results = self.model(resized, verbose=False)
                detections = results[0]

                if detections.boxes is not None:
                    for box in detections.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        confidence = float(box.conf.cpu().numpy())
                        vehicle_type = self.class_names[cls_id]

                        all_detections.append({
                            'frame_id': frame_id,
                            'vehicle_type': vehicle_type,
                            'confidence': confidence
                        })

                processed_frames += 1
                if progress_callback:
                    progress_callback(processed_frames / max_frames)

            frame_id += 1

            # Cleanup memory
            del frame
            gc.collect()

        cap.release()
        return all_detections

    def save_uploaded_video(self, uploaded_file):
        """
        Save uploaded video to a temp file for processing.
        Returns path to saved video file.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name

    def cleanup_temp_file(self, file_path):
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
