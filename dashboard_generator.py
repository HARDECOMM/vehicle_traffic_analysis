# dashboard_generator.py

import streamlit as st
import pandas as pd
import plotly.express as px

def show_key_metrics(df, processed_frames, frame_skip):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        st.metric("Unique Vehicle Types", df['vehicle_type'].nunique() if not df.empty else 0)
    with col3:
        avg_density = len(df) / processed_frames if processed_frames > 0 else 0
        st.metric(f"Avg Vehicles per {frame_skip}-th Frame", f"{avg_density:.2f}")

def plot_vehicle_type_distribution(df):
    if df.empty:
        st.warning("No detections to display in Vehicle Type Distribution.")
        return
    counts = df['vehicle_type'].value_counts().reset_index()
    counts.columns = ['Vehicle Type', 'Count']
    fig = px.bar(counts, x='Vehicle Type', y='Count', text='Count', template='plotly_white',
                 title="Vehicle Count by Type")
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def plot_vehicle_type_pie(df):
    if df.empty:
        return
    counts = df['vehicle_type'].value_counts().reset_index()
    counts.columns = ['Vehicle Type', 'Count']
    fig = px.pie(counts, names='Vehicle Type', values='Count', template='plotly_white', 
                 title="Proportion of Vehicle Types")
    st.plotly_chart(fig, use_container_width=True)

def plot_traffic_density_over_time(df, frame_skip):
    if df.empty:
        st.warning("No detections to display in Traffic Density chart.")
        return
    traffic = df.groupby('frame_id').size().reset_index(name='vehicles_detected')
    fig = px.line(traffic, x='frame_id', y='vehicles_detected', markers=True,
                  labels={"frame_id": "Frame Number", "vehicles_detected": "Vehicles Detected"},
                  title=f"Vehicles Detected per {frame_skip}-th Frame",
                  template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
