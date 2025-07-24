import cv2
import numpy as np
import streamlit as st

# ... inside your while cap.isOpened() loop, when you process a frame for inference ...

if current_frame_id % frame_skip == 0:
    status_text.text(f"Analyzing frame {current_frame_id}...")

    # Resize frame for inference as you do
    frame_for_inference = cv2.resize(frame, (320, 320))

    # Inference
    results = model(frame_for_inference, verbose=False)
    detections = results[0]

    # For drawing boxes on the original frame (not resized)
    annotated_frame = frame.copy()

    if detections.boxes is not None:
        for box in detections.boxes:
            cls_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            class_name = class_names[cls_id]
            # You can filter by confidence if needed

            # Extract box coordinates (x1, y1, x2, y2) and scale up if needed
            # Note: Your frame_for_inference is resized (320x320), so scale coords to original frame size
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            # Scale coordinates
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 320
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Draw rectangle
            color = (0, 255, 0)  # Green box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label = f"{class_name} {confidence:.2f}"
            # Put text above box
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store detection data as before if needed
            all_detections_data.append({
                'frame_id': current_frame_id,
                'vehicle_type': class_name,
                'confidence': confidence
            })

    # Show the annotated frame in Streamlit
    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
             caption=f'Frame {current_frame_id} with detections',
             use_column_width=True)

    processed_count += 1
    progress_bar.progress(min(processed_count / max_frames_to_process, 1.0))

    if processed_count >= max_frames_to_process:
        status_text.text(f"Reached maximum frames ({max_frames_to_process}). Stopping analysis.")
        break
