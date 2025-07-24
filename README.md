# vehicle_traffic_analysis# Vehicle Detection and Tracking Dashboard

## Overview

This is a **Streamlit-based vehicle detection and tracking dashboard** application using **YOLOv8 nano model** optimized for CPU inference.  
It processes uploaded traffic videos to detect and track vehicles across frames, then presents an organized dashboard with key analytics such as total unique vehicles, vehicle types distribution, and vehicle counts over time.

Designed to run efficiently on CPU-only machines (e.g., Windows 10 with Intel Core i7) with optimized performance tweaks.

---

## Features

- Upload traffic videos in mp4, avi, or mov format.
- Uses YOLOv8 nano (`yolov8n.pt`) with ByteTrack for multi-object tracking.
- Processes every 5th frame resized to 320×320 for fast CPU inference.
- Displays:
  - Total unique vehicles tracked.
  - Total detection count.
  - Vehicle type distribution bar chart.
  - Vehicle detections over time line chart.
  - Optional raw detection data in a table.

---

## Installation

1. Clone or download the repository.

2. Create and activate a Python virtual environment:

