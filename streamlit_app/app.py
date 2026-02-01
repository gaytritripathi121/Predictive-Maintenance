"""
Streamlit Dashboard for Predictive Maintenance - FINAL FIXED VERSION
Handles Keras 3.x compatibility issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import sys
sys.path.append('../src')

# Import TensorFlow with proper error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    TF_AVAILABLE = False

from data_preprocessing import CMAPSSDataLoader
from sequence_generator import SequenceGenerator
from lstm_autoencoder import LSTMAutoencoder
from anomaly_detection import AnomalyDetector


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and threshold - KERAS 3.x COMPATIBLE"""
    
    if not TF_AVAILABLE:
        st.error("TensorFlow is not available. Cannot load model.")
        return None, None, None
    
    try:
        # Possible model paths
        model_paths = [
            '../models/lstm_autoencoder.keras',
            '../models/lstm_autoencoder_best.keras',
            '../models/lstm_autoencoder.h5',
            '../models/lstm_autoencoder_best.h5',
            'models/lstm_autoencoder.keras',
            'models/lstm_autoencoder_best.keras',
        ]
        
        model = None
        loaded_path = None
        
        # Try each path
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    st.info(f"Attempting to load model from: {model_path}")
                    
                    # Load with safe mode for Keras 3.x
                    model = keras.models.load_model(
                        model_path,
                        compile=False,
                        safe_mode=False  # Important for Keras 3.x
                    )
                    loaded_path = model_path
                    st.success(f"✓ Model loaded successfully from: {model_path}")
                    break
                    
                except Exception as e:
                    st.warning(f"Failed to load from {model_path}: {str(e)[:100]}")
                    continue
        
        if model is None:
            st.error("❌ Could not load model from any path. Available files:")
            for path in model_paths:
                st.write(f"  - {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
            return None, None, None
        
        # Recompile the model
        try:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        except Exception as e:
            st.warning(f"Could not recompile model: {e}")
        
        # Load scaler
        scaler_paths = ['../models/scaler.pkl', 'models/scaler.pkl']
        scaler = None
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success(f"✓ Scaler loaded from: {scaler_path}")
                break
        
        # Load threshold
        threshold_paths = ['../models/threshold.pkl', 'models/threshold.pkl']
        threshold = None
        for threshold_path in threshold_paths:
            if os.path.exists(threshold_path):
                with open(threshold_path, 'rb') as f:
                    threshold = pickle.load(f)
                st.success(f"✓ Threshold loaded: {threshold:.6f}")
                break
        
        if scaler is None or threshold is None:
            st.error("Missing scaler or threshold files!")
            return model, None, None
        
        return model, scaler, threshold
        
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Try different data paths
        data_paths = ['../data/', 'data/', './data/']
        
        for data_path in data_paths:
            if os.path.exists(os.path.join(data_path, 'train_FD001.txt')):
                loader = CMAPSSDataLoader(data_path=data_path)
                train_df, test_df = loader.load_data()
                feature_cols, _ = loader.identify_informative_features()
                train_df = loader.split_healthy_degradation(healthy_ratio=0.35)
                return train_df, test_df, feature_cols
        
        st.error("Data files not found in any expected location")
        return None, None, None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def plot_sensor_trends_plotly(df, engine_id, feature_cols):
    """Interactive sensor trends plot"""
    engine_data = df[df['engine_id'] == engine_id]
    
    # Select top 6 most variable sensors
    sensor_variance = engine_data[feature_cols].var().sort_values(ascending=False)
    top_sensors = sensor_variance.head(6).index.tolist()
    
    fig = go.Figure()
    
    for sensor in top_sensors:
        fig.add_trace(go.Scatter(
            x=engine_data['cycle'],
            y=engine_data[sensor],
            mode='lines',
            name=sensor,
            line=dict(width=2)
        ))
    
    # Add degradation region if labels exist
    if 'label' in engine_data.columns:
        degradation = engine_data[engine_data['label'] == 1]
        if len(degradation) > 0:
            fig.add_vrect(
                x0=degradation['cycle'].min(),
                x1=degradation['cycle'].max(),
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Degradation",
                annotation_position="top left"
            )
    
    fig.update_layout(
        title=f'Sensor Trends - Engine #{engine_id}',
        xaxis_title='Cycle',
        yaxis_title='Normalized Value',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_reconstruction_error_plotly(autoencoder, detector, df, X_sequences, 
                                     metadata, engine_id):
    """Interactive reconstruction error plot"""
    # Compute errors
    errors = autoencoder.compute_reconstruction_error(X_sequences)
    
    # Filter for specific engine
    engine_mask = metadata['engine_ids'] == engine_id
    engine_cycles = metadata['cycle_indices'][engine_mask]
    engine_errors = errors[engine_mask]
    
    # Get predictions
    predictions = (engine_errors > detector.threshold).astype(int)
    
    fig = go.Figure()
    
    # Plot reconstruction error
    fig.add_trace(go.Scatter(
        x=engine_cycles,
        y=engine_errors,
        mode='lines',
        name='Reconstruction Error',
        line=dict(color='navy', width=2)
    ))
    
    # Plot threshold
    fig.add_hline(
        y=detector.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {detector.threshold:.6f}",
        annotation_position="right"
    )
    
    # Highlight anomalies
    anomaly_cycles = engine_cycles[predictions == 1]
    anomaly_errors = engine_errors[predictions == 1]
    
    if len(anomaly_cycles) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_cycles,
            y=anomaly_errors,
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    # Add degradation region
    engine_original = df[df['engine_id'] == engine_id]
    if 'label' in engine_original.columns:
        degradation = engine_original[engine_original['label'] == 1]
        if len(degradation) > 0:
            fig.add_vrect(
                x0=degradation['cycle'].min(),
                x1=degradation['cycle'].max(),
                fillcolor="orange",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="True Degradation",
                annotation_position="top left"
            )
    
    fig.update_layout(
        title=f'Reconstruction Error - Engine #{engine_id}',
        xaxis_title='Cycle',
        yaxis_title='Reconstruction Error (MSE)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">🔧 Predictive Maintenance Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("**LSTM Autoencoder for Turbofan Engine Failure Prediction**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("**Select Page:**")
    page = st.sidebar.radio("", 
                           ["📊 Overview", "🔍 Data Explorer", "🔬 Model Diagnostics", "📡 Live Monitoring"],
                           label_visibility="collapsed")
    
    # Load artifacts with progress
    with st.spinner("Loading model artifacts..."):
        model, scaler, threshold = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Cannot load model. Please check the logs above.")
        st.info("**Troubleshooting:**")
        st.markdown("""
        1. Ensure the model file exists in the `models/` directory
        2. Check TensorFlow/Keras version compatibility
        3. Try retraining the model with: `python src/train.py`
        """)
        return
    
    # Load data
    with st.spinner("Loading data..."):
        train_df, test_df, feature_cols = load_data()
    
    if train_df is None:
        st.error("⚠️ Cannot load data files.")
        return
    
    # Create autoencoder and detector
    autoencoder = LSTMAutoencoder(sequence_length=30, n_features=len(feature_cols))
    autoencoder.model = model
    
    detector = AnomalyDetector(autoencoder)
    detector.threshold = threshold
    
    # ========================================================================
    # PAGE: OVERVIEW
    # ========================================================================
    if page == "📊 Overview":
        st.header("📊 Project Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Engines", train_df['engine_id'].nunique())
            st.metric("Test Engines", test_df['engine_id'].nunique() if test_df is not None else "N/A")
        
        with col2:
            st.metric("Sensor Features", len(feature_cols))
            st.metric("Sequence Length", 30)
        
        with col3:
            st.metric("Anomaly Threshold", f"{threshold:.6f}")
            try:
                metrics_paths = ['../results/metrics.json', 'results/metrics.json']
                for metrics_path in metrics_paths:
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                        break
            except:
                st.metric("F1-Score", "N/A")
        
        st.markdown("---")
        
        # Model architecture
        st.subheader("🧠 LSTM Autoencoder Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Encoder:**")
            st.code("""
Input (30, 11)
    ↓
LSTM(64) + Dropout(0.2)
    ↓
LSTM(32) + Dropout(0.2)
    ↓
Dense(16) - Latent Space
            """)
        
        with col2:
            st.write("**Decoder:**")
            st.code("""
RepeatVector(30)
    ↓
LSTM(32) + Dropout(0.2)
    ↓
LSTM(64) + Dropout(0.2)
    ↓
TimeDistributed Dense(11)
            """)
        
        st.markdown("---")
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        try:
            metrics_paths = ['../results/metrics.json', 'results/metrics.json']
            metrics = None
            for metrics_path in metrics_paths:
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    break
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Precision", f"{metrics['precision']:.3f}",
                           help="Of predicted anomalies, how many are correct?")
                col2.metric("Recall", f"{metrics['recall']:.3f}",
                           help="Of actual anomalies, how many were detected?")
                col3.metric("F1-Score", f"{metrics['f1_score']:.3f}",
                           help="Harmonic mean of precision and recall")
                col4.metric("Accuracy", f"{metrics['accuracy']:.3f}",
                           help="Overall correctness")
            else:
                st.warning("Metrics file not found.")
                
        except Exception as e:
            st.warning(f"Could not load metrics: {e}")
    
    # ========================================================================
    # PAGE: DATA EXPLORER
    # ========================================================================
    elif page == "🔍 Data Explorer":
        st.header("🔍 Data Explorer")
        
        # Engine selector
        dataset = st.selectbox("Select Dataset", ["Training", "Test"])
        df = train_df if dataset == "Training" else test_df
        
        if df is None:
            st.warning("Test dataset not available")
            return
        
        engine_id = st.slider("Select Engine ID", 
                             min_value=int(df['engine_id'].min()),
                             max_value=int(df['engine_id'].max()),
                             value=int(df['engine_id'].min()))
        
        # Sensor trends
        st.subheader(f"Sensor Trends - Engine #{engine_id}")
        fig = plot_sensor_trends_plotly(df, engine_id, feature_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        # Engine statistics
        engine_data = df[df['engine_id'] == engine_id]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cycles", len(engine_data))
        
        if 'label' in engine_data.columns:
            col2.metric("Healthy Cycles", (engine_data['label'] == 0).sum())
            col3.metric("Degradation Cycles", (engine_data['label'] == 1).sum())
        
        # Raw data
        with st.expander("View Raw Data"):
            st.dataframe(engine_data.head(20))
    
    # ========================================================================
    # PAGE: MODEL DIAGNOSTICS
    # ========================================================================
    elif page == "🔬 Model Diagnostics":
        st.header("🔬 Model Diagnostics")
        
        # Generate sequences
        with st.spinner("Generating sequences..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            X_train_all, y_train_all, meta_train = generator.generate_sequences(
                train_df, feature_cols, label_col='label'
            )
        
        # Reconstruction error timeline
        engine_id = st.slider("Select Engine for Diagnostics",
                             min_value=1,
                             max_value=int(train_df['engine_id'].max()),
                             value=1)
        
        st.subheader(f"Reconstruction Error Timeline - Engine #{engine_id}")
        fig = plot_reconstruction_error_plotly(
            autoencoder, detector, train_df, X_train_all, meta_train, engine_id
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        st.subheader("Reconstruction Error Distribution")
        
        with st.spinner("Computing reconstruction errors..."):
            errors = autoencoder.compute_reconstruction_error(X_train_all)
        
        fig = go.Figure()
        
        # Normal distribution
        normal_errors = errors[y_train_all == 0]
        fig.add_trace(go.Histogram(
            x=normal_errors,
            name='Normal',
            marker_color='green',
            opacity=0.6,
            nbinsx=50
        ))
        
        # Anomaly distribution
        anomaly_errors = errors[y_train_all == 1]
        fig.add_trace(go.Histogram(
            x=anomaly_errors,
            name='Anomaly',
            marker_color='red',
            opacity=0.6,
            nbinsx=50
        ))
        
        # Threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Threshold: {threshold:.6f}"
        )
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title='Reconstruction Error',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE: LIVE MONITORING
    # ========================================================================
    elif page == "📡 Live Monitoring":
        st.header("📡 Live Monitoring Simulation")
        
        st.info("This simulates real-time monitoring of a turbofan engine")
        
        # Engine selector
        engine_id = st.selectbox("Select Engine to Monitor",
                                 train_df['engine_id'].unique())
        
        # Generate sequences for this engine
        with st.spinner("Loading engine data..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            X_all, y_all, meta_all = generator.generate_sequences(
                train_df, feature_cols, label_col='label'
            )
        
        # Filter for selected engine
        engine_mask = meta_all['engine_ids'] == engine_id
        X_engine = X_all[engine_mask]
        cycles_engine = meta_all['cycle_indices'][engine_mask]
        
        if len(X_engine) == 0:
            st.warning("No data available for this engine")
            return
        
        # Cycle slider
        current_cycle_idx = st.slider(
            "Current Cycle",
            min_value=0,
            max_value=len(X_engine)-1,
            value=0
        )
        
        # Get current sequence
        current_sequence = X_engine[current_cycle_idx:current_cycle_idx+1]
        current_cycle = cycles_engine[current_cycle_idx]
        
        # Compute reconstruction error
        error = autoencoder.compute_reconstruction_error(current_sequence)[0]
        is_anomaly = error > threshold
        
        # Display status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Cycle", current_cycle)
        
        with col2:
            st.metric("Reconstruction Error", f"{error:.6f}")
        
        with col3:
            status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
            color = "red" if is_anomaly else "green"
            st.markdown(f"<h3 style='color:{color}'>{status}</h3>", 
                       unsafe_allow_html=True)
        
        # Progress bar
        progress = (error / (threshold * 2)) if error < threshold * 2 else 1.0
        st.progress(progress)
        
        st.markdown("---")
        
        # Historical view
        st.subheader("Historical Error Trend")
        
        # Compute all errors up to current point
        errors_so_far = autoencoder.compute_reconstruction_error(
            X_engine[:current_cycle_idx+1]
        )
        cycles_so_far = cycles_engine[:current_cycle_idx+1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cycles_so_far,
            y=errors_so_far,
            mode='lines+markers',
            name='Reconstruction Error',
            line=dict(color='navy', width=2)
        ))
        
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold"
        )
        
        # Highlight current point
        fig.add_trace(go.Scatter(
            x=[current_cycle],
            y=[error],
            mode='markers',
            name='Current',
            marker=dict(color='yellow', size=15, symbol='star')
        ))
        
        fig.update_layout(
            xaxis_title='Cycle',
            yaxis_title='Reconstruction Error',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()