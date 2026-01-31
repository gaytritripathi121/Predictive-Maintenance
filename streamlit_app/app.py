

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from tensorflow import keras
import sys

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

# FIX: Properly add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
    """Load trained model, scaler, and threshold - FIXED VERSION"""
    try:
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        models_dir = os.path.join(parent_dir, 'models')
        
        # Try loading .keras format first (new format)
        model_path_keras = os.path.join(models_dir, 'lstm_autoencoder.keras')
        model_path_h5 = os.path.join(models_dir, 'lstm_autoencoder.h5')
        
        if os.path.exists(model_path_keras):
            model = keras.models.load_model(model_path_keras, compile=False)
            st.success("✓ Model loaded from .keras format")
        elif os.path.exists(model_path_h5):
            # Load H5 format with custom objects
            model = keras.models.load_model(model_path_h5, compile=False)
            st.warning("⚠ Loaded from .h5 format. Consider retraining to save as .keras")
        else:
            st.error("Model file not found. Please train the model first.")
            return None, None, None
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        threshold_path = os.path.join(models_dir, 'threshold.pkl')
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        
        return model, scaler, threshold
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.error("Please retrain the model using: python src/train.py")
        return None, None, None


@st.cache_data
def load_data(_scaler):
    """Load and preprocess data with proper normalization"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data') + os.sep
    
    loader = CMAPSSDataLoader(data_path=data_dir)
    train_df, test_df = loader.load_data()
    feature_cols, _ = loader.identify_informative_features()
    train_df = loader.split_healthy_degradation(healthy_ratio=0.35)
    
    # CRITICAL: Apply the loaded scaler (don't fit again!)
    train_df[feature_cols] = _scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = _scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, feature_cols


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
    st.markdown('<p class="main-header"> Predictive Maintenance Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Overview", "Data Explorer", "Model Diagnostics", "Live Monitoring"])
    
    # Load artifacts
    model, scaler, threshold = load_model_artifacts()
    
    if model is None:
        st.error(" Model artifacts not found. Please train the model first using train.py")
        st.info("Run: `python src/train.py`")
        return
    
    # Load data (pass scaler for proper normalization)
    train_df, test_df, feature_cols = load_data(scaler)
    
    # Create autoencoder and detector
    autoencoder = LSTMAutoencoder(sequence_length=30, n_features=len(feature_cols))
    autoencoder.model = model
    
    detector = AnomalyDetector(autoencoder)
    detector.threshold = threshold
    
    # ========================================================================
    # PAGE: OVERVIEW
    # ========================================================================
    if page == "Overview":
        st.header("Project Overview")
        
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
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                metrics_path = os.path.join(parent_dir, 'results', 'metrics.json')
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            except:
                st.metric("F1-Score", "N/A")
        
        st.markdown("---")
        
        # Model architecture
        st.subheader(" LSTM Autoencoder Architecture")
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
        st.subheader("Performance Metrics")
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            metrics_path = os.path.join(parent_dir, 'results', 'metrics.json')
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
            
        except:
            st.warning("Metrics file not found. Train the model first.")
    
    # ========================================================================
    # PAGE: DATA EXPLORER
    # ========================================================================
    elif page == "Data Explorer":
        st.header("🔍 Data Explorer")
        
        # Engine selector
        dataset = st.selectbox("Select Dataset", ["Training", "Test"])
        df = train_df if dataset == "Training" else test_df
        
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
    elif page == "Model Diagnostics":
        st.header(" Model Diagnostics")
        
        # Generate sequences
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
        
        errors = autoencoder.compute_reconstruction_error(X_train_all)
        predictions = (errors > threshold).astype(int)
        
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
    elif page == "Live Monitoring":
        st.header("📡 Live Monitoring Simulation")
        
        st.info("This simulates real-time monitoring of a turbofan engine")
        
        # Add explanation
        st.markdown("""
        **Understanding the Status:**
        - 🟢 **Early cycles (30-70)**: Healthy operation → Status will be  NORMAL
        - 🔴 **Late cycles (120+)**: Degradation begins → Status will be  ANOMALY
        - Use the slider below to simulate different points in the engine's lifecycle
        """)
        
        st.markdown("---")
        
        # Engine selector
        engine_id = st.selectbox("Select Engine to Monitor",
                                 sorted(train_df['engine_id'].unique()))
        
        # Generate sequences for this engine
        generator = SequenceGenerator(sequence_length=30, stride=1)
        X_all, y_all, meta_all = generator.generate_sequences(
            train_df, feature_cols, label_col='label'
        )
        
        # Filter for selected engine
        engine_mask = meta_all['engine_ids'] == engine_id
        X_engine = X_all[engine_mask]
        cycles_engine = meta_all['cycle_indices'][engine_mask]
        y_engine = y_all[engine_mask]
        
        if len(X_engine) == 0:
            st.warning("No data available for this engine")
            return
        
        # Get engine info
        engine_data_full = train_df[train_df['engine_id'] == engine_id]
        max_cycle = engine_data_full['cycle'].max()
        healthy_threshold_cycle = int(max_cycle * 0.35)
        
        # Show available cycle range
        min_available_cycle = int(cycles_engine.min())
        max_available_cycle = int(cycles_engine.max())
        
        st.info(f"**Engine #{engine_id} Info:** Cycles {min_available_cycle}-{max_available_cycle} available (sequences need 30 cycles of history)")
        
        # Let user select by ACTUAL CYCLE NUMBER
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_cycle = st.slider(
                "Select Actual Cycle Number",
                min_value=min_available_cycle,
                max_value=max_available_cycle,
                value=min_available_cycle,
                help="Select the engine cycle you want to monitor"
            )
        
        with col2:
            st.write("**Quick Jump:**")
            if st.button("🟢 Early"):
                selected_cycle = min_available_cycle
                st.rerun()
            if st.button("🟡 Mid"):
                selected_cycle = (min_available_cycle + max_available_cycle) // 2
                st.rerun()
            if st.button("🔴 Late"):
                selected_cycle = max_available_cycle
                st.rerun()
        
        # Find the closest sequence index for the selected cycle
        cycle_diffs = np.abs(cycles_engine - selected_cycle)
        current_cycle_idx = int(np.argmin(cycle_diffs))
        current_cycle = int(cycles_engine[current_cycle_idx])
        
        # Get current data
        current_sequence = X_engine[current_cycle_idx:current_cycle_idx+1]
        true_label = y_engine[current_cycle_idx]
        
        # Compute reconstruction error
        error = autoencoder.compute_reconstruction_error(current_sequence)[0]
        
        # Sanity check for error value
        if error > 100:
            st.error(f" Abnormal reconstruction error detected: {error:.2f}")
            st.error("This suggests a data normalization issue. Please check that:")
            st.code("""
1. The model was trained with the same scaler
2. Data is properly normalized
3. Scaler.pkl matches the trained model
            """)
            st.warning("Trying to continue anyway, but results may be unreliable...")
        
        is_anomaly = error > threshold
        
        # Display status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Selected Cycle", int(current_cycle))
            st.caption(f"Sequence Index: {current_cycle_idx}")
            lifecycle_pct = (current_cycle / max_cycle) * 100
            st.caption(f" {lifecycle_pct:.1f}% of lifecycle")
        
        with col2:
            # Show error with better formatting
            if error < 10:
                st.metric("Reconstruction Error", f"{error:.4f}")
            else:
                st.metric("Reconstruction Error", f"{error:.2e}")  # Scientific notation for large values
            st.caption(f"Threshold: {threshold:.4f}")
        
        with col3:
            true_status = "🟢 Healthy" if true_label == 0 else "🔴 Degrading"
            st.markdown("**True State:**")
            st.markdown(f"<h4>{true_status}</h4>", unsafe_allow_html=True)
        
        with col4:
            status = " ANOMALY" if is_anomaly else "NORMAL"
            color = "red" if is_anomaly else "green"
            st.markdown("**Model Says:**")
            st.markdown(f"<h3 style='color:{color}'>{status}</h3>", 
                       unsafe_allow_html=True)
        
        # Show agreement
        agrees = (is_anomaly and true_label == 1) or (not is_anomaly and true_label == 0)
        if agrees:
            st.success("✓ Model prediction matches true state!")
        else:
            if is_anomaly and true_label == 0:
                st.warning(" False Positive: Model flags healthy cycle as anomaly")
            else:
                st.error("False Negative: Model missed degradation!")
        
        # Progress bar
        progress = min((error / threshold), 1.0)
        st.progress(progress)
        
        if error < threshold * 0.5:
            st.success("🟢 Low Risk - Normal Operation")
        elif error < threshold:
            st.info("🟡 Medium Risk - Monitor Closely")
        elif error < threshold * 1.5:
            st.warning("🟠 High Risk - Schedule Inspection")
        else:
            st.error("🔴 Critical Risk - Immediate Action Required")
        
        st.markdown("---")
        
        # Historical view
        st.subheader("📈 Historical Error Trend")
        
        # Compute ALL errors for this engine
        all_errors = autoencoder.compute_reconstruction_error(X_engine)
        
        fig = go.Figure()
        
        # Plot all errors
        fig.add_trace(go.Scatter(
            x=cycles_engine,
            y=all_errors,
            mode='lines',
            name='Reconstruction Error',
            line=dict(color='navy', width=2)
        ))
        
        # Add threshold
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Anomaly Threshold"
        )
        
        # Add healthy/degradation regions
        fig.add_vrect(
            x0=0,
            x1=healthy_threshold_cycle,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="Healthy Region",
            annotation_position="top left"
        )
        
        fig.add_vrect(
            x0=healthy_threshold_cycle,
            x1=max_cycle,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="Degradation Region",
            annotation_position="top right"
        )
        
        # Highlight current point
        fig.add_trace(go.Scatter(
            x=[current_cycle],
            y=[error],
            mode='markers',
            name='Current Position',
            marker=dict(color='yellow', size=20, symbol='star', 
                       line=dict(color='black', width=2))
        ))
        
        fig.update_layout(
            xaxis_title='Actual Cycle Number',
            yaxis_title='Reconstruction Error (MSE)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation guide
        with st.expander(" How to Interpret This Dashboard"):
            st.markdown(f"""
            ### Understanding the Model
            
            **Engine #{engine_id} Info:**
            - Total lifecycle: {max_cycle} cycles
            - Healthy region: Cycles 1-{healthy_threshold_cycle}
            - Degradation region: Cycles {healthy_threshold_cycle+1}-{max_cycle}
            - First sequence starts at cycle 30 (needs 30 cycles of history)
            
            **How It Works:**
            1. Model was trained ONLY on healthy data (first 35% of engine life)
            2. Low error (< {threshold:.4f}) = Recognized pattern = NORMAL 
            3. High error (> {threshold:.4f}) = Unknown pattern = ANOMALY 
            
            **To See Both States:**
            - Move slider LEFT (index 0-50) → See NORMAL status on early cycles
            - Move slider RIGHT (index 100+) → See ANOMALY status on late cycles
            - Watch error increase as engine degrades!
            """)


if __name__ == "__main__":
    main()