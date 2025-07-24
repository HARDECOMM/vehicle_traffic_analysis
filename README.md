## Vehicle Traffic Analysis Dashboard

This project is a vehicle traffic analysis dashboard built with Streamlit and YOLOv8 deep learning model that detects vehicles in uploaded traffic videos. It processes video frames to detect different types of vehicles, then visualizes insightful metrics and charts interactively for easy analysis.

The app is split into 3 modular parts for maintainability and scalability:

    - vehicle_analyzer.py: Handles the video processing and vehicle detection using the YOLOv8 model.

    - dashboard_generator.py: Generates interactive KPIs and charts from the detection results using Streamlit and Plotly.

    - main.py: Main Streamlit app, integrating UI components, file upload, parameter controls, running the analysis, and showing results.

## Features

    Upload traffic videos (MP4, AVI, MOV)

    Tune analysis parameters:

        - Process every N-th frame (frame skipping)

        - Maximum number of frames to analyze

    High-performance, lightweight YOLOv8 model for fast vehicle detection

    Interactive dashboard with:

        Total detections and unique vehicle types metrics

        Vehicle type distribution bar chart

        Vehicle type proportion pie chart

        Traffic density over time line chart

    Downloadable CSV reports for full detections and summaries

## Installation
#### Requirements

    Python 3.11 specify on config.toml

    Required Python libraries listed in requirements.txt (to be created)

## Install via pip:

bash
pip install streamlit ultralytics opencv-python-headless pandas plotly

Make sure to install opencv-python-headless to avoid GUI-related dependency issues on servers or cloud platforms.
Usage

    Clone this repository (or download the three .py files):

bash
git clone <your-repo-url>
cd <repo-folder>

    Run the Streamlit app:

bash
streamlit run main.py

    Open the app in your browser (usually at http://localhost:8501).

    Upload a traffic video (supported formats: .mp4, .avi, .mov).

    Adjust sidebar parameters for frame skipping and max frames to analyze for speed/accuracy trade-off.

    Click "Start Analysis" to process the video.

    Explore dashboard visuals and download CSV reports.

## Module Details
#### vehicle_analyzer.py

    - Loads YOLOv8 model (yolov8n.pt) with caching for fast repeated use.

    - Saves uploaded videos as temporary files for reliable processing.

    - Processes video frames, respecting frame_skip and max_frames.
 
    - Extracts vehicle detection info (frame number, vehicle type, confidence).

    - Returns detection results as a list of dictionaries for further use.

#### dashboard_generator.py

    Accepts detection results as a pandas DataFrame.

    Displays key metrics like total vehicle detections, unique types, and average detections per frame.

    Generates interactive visual charts:

        - Bar chart showing vehicle counts by type.

        - Pie chart showing vehicle type proportions.

        - Line chart showing traffic density over processed frames.

    Uses Plotly for smooth, interactive visualizations integrated into Streamlit.

#### main.py

    Contains the Streamlit UI:

        Video file upload with session state persistence.

        Sidebar controls for tuning frame skip and max frames analyzed.

        Progress bar and status messaging during analysis.

    Calls the analyzer module to run detection.

    Passes results to dashboard generator to produce visuals.

    Provides CSV download buttons for data extraction.

    Handles cleanup of temporary files and manages app state cleanly.

# Tips for Best Results

    Choose lower frame skipping values and higher max frames for more 

    Use well-lit videos to increase performance.

    Monitor sidebar parameters to balance speed versus detection detail.

    Consider deploying on platforms supporting GPU acceleration for faster YOLO execution.

# License

This project is open-source and free to use for educational and research purposes.
Contact

Feel free to submit issues or pull requests, or contact me at:

    GitHub: https://github.com/HARDECOMM

    Email: ademoyeharuna@gmail.com

Happy vehicle detection and analysis! 🚗📊

If you want, I can also help you write a requirements.txt and/or a sample .gitignore for this project.