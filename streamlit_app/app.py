"""
Streamlit Dashboard for Predictive Maintenance - PRODUCTION VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import plotly.graph_objects as go
from tensorflow import keras

# ============================================================================
# PATH CONFIGURATION - Works on Streamlit Cloud and locally
# ============================================================================

# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Get project root (parent of streamlit_app)
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Add src to Python path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Define paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

# Now import local modules
try:
    from data_preprocessing import CMAPSSDataLoader
    from sequence_generator import SequenceGenerator
    from lstm_autoencoder import LSTMAutoencoder
    from anomaly_detection import AnomalyDetector
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.info("Make sure all source files are in the 'src/' directory")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

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
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and threshold"""
    try:
        # Try loading .keras format first
        model_path_keras = os.path.join(MODEL_PATH, 'lstm_autoencoder.keras')
        model_path_h5 = os.path.join(MODEL_PATH, 'lstm_autoencoder.h5')
        
        if os.path.exists(model_path_keras):
            model = keras.models.load_model(model_path_keras, compile=False)
            st.sidebar.success("✓ Model loaded (.keras)")
        elif os.path.exists(model_path_h5):
            model = keras.models.load_model(model_path_h5, compile=False)
            st.sidebar.success("✓ Model loaded (.h5)")
        else:
            st.error(f"❌ Model not found in: {MODEL_PATH}")
            st.info("Please add model files to the repository")
            return None, None, None
        
        # Recompile
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load scaler and threshold
        scaler_path = os.path.join(MODEL_PATH, 'scaler.pkl')
        threshold_path = os.path.join(MODEL_PATH, 'threshold.pkl')
        
        if not os.path.exists(scaler_path) or not os.path.exists(threshold_path):
            st.error("❌ Scaler or threshold file not found")
            return None, None, None
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        
        return model, scaler, threshold
        
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        with st.expander("Show full error"):
            st.exception(e)
        return None, None, None


@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        loader = CMAPSSDataLoader(data_path=DATA_PATH)
        train_df, test_df = loader.load_data()
        feature_cols, _ = loader.identify_informative_features()
        train_df = loader.split_healthy_degradation(healthy_ratio=0.35)
        return train_df, test_df, feature_cols
    except FileNotFoundError as e:
        st.error("❌ Dataset files not found!")
        st.info(f"Looking in: {DATA_PATH}")
        st.info("Please add train_FD001.txt and test_FD001.txt to data/ folder")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        with st.expander("Show full error"):
            st.exception(e)
        return None, None, None


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

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


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">🔧 Predictive Maintenance Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("**LSTM Autoencoder for Turbofan Engine Failure Prediction**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio("Select Page:", 
                           ["📊 Overview", "🔍 Data Explorer", "🔬 Model Diagnostics", "📡 Live Monitoring"])
    
    # Load artifacts with error handling
    with st.spinner("Loading model and data..."):
        model, scaler, threshold = load_model_artifacts()
        
        if model is None:
            st.error("⚠️ Cannot load model. Please check the logs above.")
            st.stop()
        
        train_df, test_df, feature_cols = load_data()
        
        if train_df is None:
            st.error("⚠️ Cannot load data. Please check the logs above.")
            st.stop()
    
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
            st.metric("Test Engines", test_df['engine_id'].nunique())
        
        with col2:
            st.metric("Sensor Features", len(feature_cols))
            st.metric("Sequence Length", 30)
        
        with col3:
            st.metric("Anomaly Threshold", f"{threshold:.6f}")
            try:
                metrics_path = os.path.join(RESULTS_PATH, 'metrics.json')
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
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
            metrics_path = os.path.join(RESULTS_PATH, 'metrics.json')
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Precision", f"{metrics['precision']:.3f}",
                       help="Of predicted anomalies, how many are correct?")
            col2.metric("Recall", f"{metrics['recall']:.3f}",
                       help="Of actual anomalies, how many were detected?")
            col3.metric("F1-Score", f"{metrics['f1_score']:.3f}",
                       help="Harmonic mean of precision and recall")
            col4.metric("Accuracy", f"{metrics['accuracy']:.3f}",
                       help="Overall correctness")
            
        except Exception as e:
            st.warning("⚠️ Metrics file not found. Model may need retraining.")
    
    # ========================================================================
    # PAGE: DATA EXPLORER
    # ========================================================================
    elif page == "🔍 Data Explorer":
        st.header("🔍 Data Explorer")
        
        dataset = st.selectbox("Select Dataset", ["Training", "Test"])
        df = train_df if dataset == "Training" else test_df
        
        engine_id = st.slider("Select Engine ID", 
                             min_value=int(df['engine_id'].min()),
                             max_value=int(df['engine_id'].max()),
                             value=int(df['engine_id'].min()))
        
        st.subheader(f"Sensor Trends - Engine #{engine_id}")
        fig = plot_sensor_trends_plotly(df, engine_id, feature_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        engine_data = df[df['engine_id'] == engine_id]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cycles", len(engine_data))
        
        if 'label' in engine_data.columns:
            col2.metric("Healthy Cycles", (engine_data['label'] == 0).sum())
            col3.metric("Degradation Cycles", (engine_data['label'] == 1).sum())
        
        with st.expander("View Raw Data"):
            st.dataframe(engine_data.head(20))
    
    # ========================================================================
    # PAGE: MODEL DIAGNOSTICS
    # ========================================================================
    elif page == "🔬 Model Diagnostics":
        st.header("🔬 Model Diagnostics")
        
        with st.spinner("Generating sequences..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            X_train_all, y_train_all, meta_train = generator.generate_sequences(
                train_df, feature_cols, label_col='label'
            )
        
        engine_id = st.slider("Select Engine for Diagnostics",
                             min_value=1,
                             max_value=int(train_df['engine_id'].max()),
                             value=1)
        
        st.subheader(f"Reconstruction Error Timeline - Engine #{engine_id}")
        fig = plot_reconstruction_error_plotly(
            autoencoder, detector, train_df, X_train_all, meta_train, engine_id
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Reconstruction Error Distribution")
        
        errors = autoencoder.compute_reconstruction_error(X_train_all)
        
        fig = go.Figure()
        
        normal_errors = errors[y_train_all == 0]
        fig.add_trace(go.Histogram(
            x=normal_errors,
            name='Normal',
            marker_color='green',
            opacity=0.6,
            nbinsx=50
        ))
        
        anomaly_errors = errors[y_train_all == 1]
        fig.add_trace(go.Histogram(
            x=anomaly_errors,
            name='Anomaly',
            marker_color='red',
            opacity=0.6,
            nbinsx=50
        ))
        
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
        
        engine_id = st.selectbox("Select Engine to Monitor",
                                 train_df['engine_id'].unique())
        
        with st.spinner("Generating sequences..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            X_all, y_all, meta_all = generator.generate_sequences(
                train_df, feature_cols, label_col='label'
            )
        
        engine_mask = meta_all['engine_ids'] == engine_id
        X_engine = X_all[engine_mask]
        cycles_engine = meta_all['cycle_indices'][engine_mask]
        
        if len(X_engine) == 0:
            st.warning("No data available for this engine")
            return
        
        current_cycle_idx = st.slider(
            "Current Cycle",
            min_value=0,
            max_value=len(X_engine)-1,
            value=0
        )
        
        current_sequence = X_engine[current_cycle_idx:current_cycle_idx+1]
        current_cycle = cycles_engine[current_cycle_idx]
        
        error = autoencoder.compute_reconstruction_error(current_sequence)[0]
        is_anomaly = error > threshold
        
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
        
        progress = min((error / (threshold * 2)), 1.0)
        st.progress(progress)
        
        st.markdown("---")
        st.subheader("Historical Error Trend")
        
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