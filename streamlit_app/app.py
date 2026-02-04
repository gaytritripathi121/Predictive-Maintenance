"""
Streamlit Dashboard for Predictive Maintenance - PRODUCTION VERSION
FINAL STABLE VERSION with comprehensive NumPy compatibility fixes
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
# COMPREHENSIVE NUMPY COMPATIBILITY FIX - APPLY BEFORE ANY IMPORTS
# ============================================================================

def fix_numpy_compatibility():
    """
    Comprehensive fix for NumPy version compatibility issues.
    Handles both numpy._core and numpy.core._multiarray_umath errors.
    Must be called BEFORE loading any pickled objects.
    """
    import numpy as np
    import sys
    
    # Fix 1: numpy._core compatibility
    if not hasattr(np, '_core'):
        np._core = np.core
    
    # Fix 2: Create missing submodules
    if not hasattr(np.core, '_multiarray_umath'):
        try:
            # Try to import it properly
            from numpy.core import _multiarray_umath
            np.core._multiarray_umath = _multiarray_umath
        except ImportError:
            # If not available, create a mock
            class MockMultiarrayUmath:
                pass
            np.core._multiarray_umath = MockMultiarrayUmath()
    
    # Fix 3: Update sys.modules to recognize the patched modules
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, '_multiarray_umath'):
        sys.modules['numpy.core._multiarray_umath'] = np.core._multiarray_umath

# Apply fix immediately
fix_numpy_compatibility()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

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
# ROBUST PICKLE LOADING WITH MULTIPLE FALLBACK STRATEGIES
# ============================================================================

def safe_pickle_load(filepath, max_retries=3):
    """
    Safely load pickle files with comprehensive error handling.
    Uses multiple fallback strategies for maximum compatibility.
    
    Args:
        filepath: Path to pickle file
        max_retries: Number of retry attempts with different strategies
    
    Returns:
        Unpickled object
    """
    import pickle
    import numpy as np
    
    # Ensure NumPy compatibility is applied
    fix_numpy_compatibility()
    
    for attempt in range(max_retries):
        try:
            # Strategy 1: Standard pickle load
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            return obj
            
        except ModuleNotFoundError as e:
            error_msg = str(e)
            
            # Strategy 2: Handle numpy._core errors
            if 'numpy._core' in error_msg or 'numpy.core' in error_msg:
                if attempt == 0:
                    st.sidebar.warning(f"⚠️ NumPy compatibility issue detected (attempt {attempt + 1}/{max_retries})")
                
                # Reapply fix
                fix_numpy_compatibility()
                
                # Try using encoding parameter
                try:
                    with open(filepath, 'rb') as f:
                        obj = pickle.load(f, encoding='latin1')
                    st.sidebar.success("✓ Loaded with latin1 encoding")
                    return obj
                except:
                    pass
                
                # Try using different protocol
                try:
                    import pickle5
                    with open(filepath, 'rb') as f:
                        obj = pickle5.load(f)
                    st.sidebar.success("✓ Loaded with pickle5")
                    return obj
                except:
                    pass
                
                if attempt < max_retries - 1:
                    continue
                else:
                    raise Exception(
                        f"Failed to load {filepath} after {max_retries} attempts. "
                        f"This file was likely created with a different NumPy version. "
                        f"Please regenerate it by running train.py"
                    )
            else:
                raise
                
        except Exception as e:
            if attempt < max_retries - 1:
                st.sidebar.warning(f"⚠️ Retry {attempt + 1}/{max_retries}: {str(e)[:50]}...")
                continue
            else:
                raise Exception(f"Failed to load {filepath}: {str(e)}")
    
    raise Exception(f"Failed to load {filepath} after {max_retries} attempts")


def safe_joblib_load(filepath):
    """
    Safely load joblib files with NumPy compatibility.
    """
    import joblib
    
    # Ensure NumPy compatibility
    fix_numpy_compatibility()
    
    try:
        return joblib.load(filepath)
    except Exception as e:
        st.error(f"Error loading joblib file: {str(e)}")
        st.info("Please regenerate the file by running train.py")
        raise


# ============================================================================
# ROBUST MODEL LOADING WITH MULTIPLE FALLBACK METHODS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """
    Load model with multiple fallback methods for maximum compatibility.
    Handles both Keras model loading and NumPy pickle compatibility.
    """
    
    def try_load_saved_model():
        """Try loading saved model directly"""
        paths = [
            os.path.join(MODEL_PATH, 'lstm_autoencoder.keras'),
            os.path.join(MODEL_PATH, 'lstm_autoencoder_best.keras'),
            os.path.join(MODEL_PATH, 'lstm_autoencoder.h5')
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    model = keras.models.load_model(path, compile=False, safe_mode=False)
                    st.sidebar.success(f"✓ Model loaded: {os.path.basename(path)}")
                    return model
                except Exception as e:
                    st.sidebar.warning(f"⚠️ {os.path.basename(path)}: {str(e)[:30]}...")
        return None
    
    def rebuild_and_load_weights():
        """Rebuild architecture and load weights"""
        try:
            params_path = os.path.join(MODEL_PATH, 'model_params.json')
            
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                sequence_length = params['sequence_length']
                n_features = params['n_features']
                latent_dim = params['latent_dim']
                st.sidebar.info(f"📄 params: seq={sequence_length}, feat={n_features}")
            else:
                sequence_length, n_features, latent_dim = 30, 11, 16
                st.sidebar.warning("⚠️ Using default params")
            
            autoencoder = LSTMAutoencoder(sequence_length, n_features, latent_dim)
            model = autoencoder.build_model()
            
            weights_path = os.path.join(MODEL_PATH, 'lstm_autoencoder.weights.h5')
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                st.sidebar.success("✓ Rebuilt & weights loaded")
                return model
            else:
                st.sidebar.error("❌ Weights file not found")
                return None
                
        except Exception as e:
            st.sidebar.error(f"❌ Rebuild failed: {str(e)[:40]}...")
            return None
    
    try:
        st.sidebar.subheader("🔄 Loading Model...")
        
        # Step 1: Load model
        model = try_load_saved_model()
        if model is None:
            st.sidebar.info("Trying alternative method...")
            model = rebuild_and_load_weights()
        
        if model is None:
            st.error(f"❌ Could not load model from: {MODEL_PATH}")
            st.info("**Required files in models/ directory:**")
            st.code("• lstm_autoencoder.keras\n• lstm_autoencoder.weights.h5\n• model_params.json\n• scaler.pkl (or scaler.joblib)\n• threshold.pkl")
            st.info("💡 Run `train.py` to generate these files")
            st.stop()
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Step 2: Load scaler with comprehensive fallback
        scaler = None
        scaler_paths = [
            ('scaler.joblib', safe_joblib_load),
            ('scaler.pkl', safe_pickle_load)
        ]
        
        for filename, load_func in scaler_paths:
            scaler_path = os.path.join(MODEL_PATH, filename)
            if os.path.exists(scaler_path):
                try:
                    st.sidebar.info(f"Loading {filename}...")
                    scaler = load_func(scaler_path)
                    st.sidebar.success(f"✓ Scaler loaded from {filename}")
                    break
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Failed to load {filename}: {str(e)[:40]}...")
                    continue
        
        if scaler is None:
            st.error("❌ Could not load scaler from any available format")
            st.info("**Tried:**")
            st.code("• scaler.joblib\n• scaler.pkl")
            st.info("💡 Run `train.py` to regenerate scaler")
            
            with st.expander("📋 Troubleshooting Steps"):
                st.markdown("""
                **This error occurs due to NumPy version mismatch. To fix:**
                
                1. **Quick Fix (Recommended)**: Regenerate the scaler
                   ```bash
                   python train.py
                   ```
                
                2. **Alternative**: Upgrade NumPy to match training environment
                   ```bash
                   pip install --upgrade numpy
                   ```
                
                3. **Check NumPy version**:
                   - Current: `{np.__version__}`
                   - Required: Check your training environment
                
                4. **If still failing**: Delete old pickle files and retrain
                   ```bash
                   rm models/scaler.pkl models/threshold.pkl
                   python train.py
                   ```
                """)
            st.stop()
        
        # Step 3: Load threshold
        threshold_path = os.path.join(MODEL_PATH, 'threshold.pkl')
        
        if not os.path.exists(threshold_path):
            st.error("❌ Threshold file not found")
            st.info(f"Expected: {threshold_path}")
            st.info("💡 Run `train.py` to generate threshold.pkl")
            st.stop()
        
        try:
            threshold = safe_pickle_load(threshold_path)
            st.sidebar.success(f"✓ Threshold: {threshold:.6f}")
        except Exception as e:
            st.error(f"❌ Error loading threshold: {str(e)}")
            st.info("💡 Run `train.py` to regenerate threshold.pkl")
            st.stop()
        
        with st.sidebar.expander("📊 Model Info"):
            st.text(f"Input: {model.input_shape}")
            st.text(f"Params: {model.count_params():,}")
        
        st.sidebar.success("✅ All artifacts loaded successfully")
        return model, scaler, threshold
        
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("Full Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("**Recovery Steps:**")
        st.code("1. Delete models/*.pkl files\n2. Run: python train.py\n3. Restart dashboard")
        st.stop()


@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        data_paths = ['data/', '../data/', './data/', os.path.join(os.getcwd(), 'data/')]
        
        for data_path in data_paths:
            train_file = os.path.join(data_path, 'train_FD001.txt')
            test_file = os.path.join(data_path, 'test_FD001.txt')
            
            if os.path.exists(train_file) and os.path.exists(test_file):
                st.sidebar.success(f"✓ Data loaded from: {data_path}")
                loader = CMAPSSDataLoader(data_path=data_path)
                train_df, test_df = loader.load_data()
                feature_cols, _ = loader.identify_informative_features()
                train_df = loader.split_healthy_degradation(healthy_ratio=0.35)
                return train_df, test_df, feature_cols
        
        st.error("❌ Dataset files not found!")
        st.info("Please ensure data files are in the 'data/' directory")
        return None, None, None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_data
def scale_dataframe(_scaler, df, feature_cols):
    """Scale features"""
    df_scaled = df.copy()
    df_scaled[feature_cols] = _scaler.transform(df[feature_cols])
    return df_scaled


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_sensor_trends_plotly(df, engine_id, feature_cols):
    """Interactive sensor trends"""
    engine_data = df[df['engine_id'] == engine_id]
    sensor_variance = engine_data[feature_cols].var().sort_values(ascending=False)
    top_sensors = sensor_variance.head(6).index.tolist()
    
    fig = go.Figure()
    for sensor in top_sensors:
        fig.add_trace(go.Scatter(
            x=engine_data['cycle'], y=engine_data[sensor],
            mode='lines', name=sensor, line=dict(width=2)
        ))
    
    if 'label' in engine_data.columns:
        degradation = engine_data[engine_data['label'] == 1]
        if len(degradation) > 0:
            fig.add_vrect(
                x0=degradation['cycle'].min(), x1=degradation['cycle'].max(),
                fillcolor="red", opacity=0.2, layer="below", line_width=0,
                annotation_text="Degradation", annotation_position="top left"
            )
    
    fig.update_layout(
        title=f'Sensor Trends - Engine #{engine_id}',
        xaxis_title='Cycle', yaxis_title='Normalized Value',
        hovermode='x unified', height=500, template='plotly_white'
    )
    return fig


def plot_reconstruction_error_plotly(autoencoder, detector, df, X_sequences, metadata, engine_id):
    """Interactive reconstruction error"""
    errors = autoencoder.compute_reconstruction_error(X_sequences)
    engine_mask = metadata['engine_ids'] == engine_id
    engine_cycles = metadata['cycle_indices'][engine_mask]
    engine_errors = errors[engine_mask]
    predictions = (engine_errors > detector.threshold).astype(int)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=engine_cycles, y=engine_errors, mode='lines',
        name='Reconstruction Error', line=dict(color='navy', width=2)
    ))
    
    fig.add_hline(
        y=detector.threshold, line_dash="dash", line_color="red",
        annotation_text=f"Threshold: {detector.threshold:.6f}", annotation_position="right"
    )
    
    anomaly_cycles = engine_cycles[predictions == 1]
    anomaly_errors = engine_errors[predictions == 1]
    
    if len(anomaly_cycles) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_cycles, y=anomaly_errors, mode='markers',
            name='Detected Anomalies', marker=dict(color='red', size=8, symbol='x')
        ))
    
    engine_original = df[df['engine_id'] == engine_id]
    if 'label' in engine_original.columns:
        degradation = engine_original[engine_original['label'] == 1]
        if len(degradation) > 0:
            fig.add_vrect(
                x0=degradation['cycle'].min(), x1=degradation['cycle'].max(),
                fillcolor="orange", opacity=0.15, layer="below", line_width=0,
                annotation_text="True Degradation", annotation_position="top left"
            )
    
    fig.update_layout(
        title=f'Reconstruction Error - Engine #{engine_id}',
        xaxis_title='Cycle', yaxis_title='Reconstruction Error (MSE)',
        hovermode='x unified', height=500, template='plotly_white'
    )
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<p class="main-header">🔧 Predictive Maintenance Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**LSTM Autoencoder for Turbofan Engine Failure Prediction**")
    st.markdown("---")
    
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio("Select Page:", 
                           ["📊 Overview", "🔍 Data Explorer", "🔬 Model Diagnostics", "📡 Live Monitoring"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    with st.spinner("Loading model artifacts..."):
        model, scaler, threshold = load_model_artifacts()
    
    with st.spinner("Loading data..."):
        train_df, test_df, feature_cols = load_data()
        if train_df is None:
            st.error("⚠️ Cannot load data")
            st.stop()
    
    autoencoder = LSTMAutoencoder(sequence_length=30, n_features=len(feature_cols))
    autoencoder.model = model
    
    detector = AnomalyDetector(autoencoder)
    detector.threshold = threshold
    
    # OVERVIEW PAGE
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
                with open(os.path.join(RESULTS_PATH, 'metrics.json'), 'r') as f:
                    metrics = json.load(f)
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            except:
                st.metric("F1-Score", "N/A")
        
        st.markdown("---")
        st.subheader("🧠 LSTM Autoencoder Architecture")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Encoder:**")
            st.code("Input (30,11)\n  ↓\nLSTM(64)+Dropout\n  ↓\nLSTM(32)+Dropout\n  ↓\nDense(16)")
        with col2:
            st.write("**Decoder:**")
            st.code("RepeatVector(30)\n  ↓\nLSTM(32)+Dropout\n  ↓\nLSTM(64)+Dropout\n  ↓\nDense(11)")
        
        st.markdown("---")
        st.subheader("📈 Performance Metrics")
        try:
            with open(os.path.join(RESULTS_PATH, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", f"{metrics['precision']:.3f}")
            col2.metric("Recall", f"{metrics['recall']:.3f}")
            col3.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            col4.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        except:
            st.warning("⚠️ Metrics file not found")
        
        st.markdown("---")
        st.subheader("📊 Selected Sensor Features")
        cols = st.columns(3)
        for idx, feature in enumerate(feature_cols):
            cols[idx % 3].write(f"• {feature}")
    
    # DATA EXPLORER PAGE
    elif page == "🔍 Data Explorer":
        st.header("🔍 Data Explorer")
        
        dataset = st.selectbox("Select Dataset", ["Training", "Test"])
        df = train_df if dataset == "Training" else test_df
        
        engine_id = st.slider("Select Engine ID", 
                             int(df['engine_id'].min()), int(df['engine_id'].max()),
                             int(df['engine_id'].min()))
        
        st.subheader(f"Sensor Trends - Engine #{engine_id}")
        fig = plot_sensor_trends_plotly(df, engine_id, feature_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        engine_data = df[df['engine_id'] == engine_id]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cycles", len(engine_data))
        if 'label' in engine_data.columns:
            col2.metric("Healthy Cycles", (engine_data['label'] == 0).sum())
            col3.metric("Degradation Cycles", (engine_data['label'] == 1).sum())
        
        st.markdown("---")
        st.subheader(f"📊 Engine #{engine_id} Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sensor Value Ranges**")
            st.dataframe(engine_data[feature_cols].describe().T[['min', 'max', 'mean', 'std']])
        with col2:
            st.write("**Sensor Variability**")
            variance_df = pd.DataFrame({
                'Sensor': feature_cols,
                'Variance': engine_data[feature_cols].var().values
            }).sort_values('Variance', ascending=False)
            st.dataframe(variance_df)
        
        with st.expander("📋 View Raw Data"):
            st.dataframe(engine_data.head(50))
    
    # MODEL DIAGNOSTICS PAGE
    elif page == "🔬 Model Diagnostics":
        st.header("🔬 Model Diagnostics")
        st.info("⚙️ Analyzing reconstruction errors for anomaly detection")
        
        with st.spinner("Generating sequences..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            train_df_scaled = scale_dataframe(scaler, train_df, feature_cols)
            X_train_all, y_train_all, meta_train = generator.generate_sequences(
                train_df_scaled, feature_cols, label_col='label'
            )
        
        engine_id = st.slider("Select Engine", 1, int(train_df['engine_id'].max()), 1)
        
        st.subheader(f"Reconstruction Error - Engine #{engine_id}")
        fig = plot_reconstruction_error_plotly(
            autoencoder, detector, train_df, X_train_all, meta_train, engine_id
        )
        st.plotly_chart(fig, use_container_width=True)
        
        engine_mask = meta_train['engine_ids'] == engine_id
        engine_errors = autoencoder.compute_reconstruction_error(X_train_all[engine_mask])
        engine_labels = y_train_all[engine_mask]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Error", f"{engine_errors.mean():.6f}")
        col2.metric("Max Error", f"{engine_errors.max():.6f}")
        col3.metric("Detected Anomalies", (engine_errors > threshold).sum())
        if 'label' in train_df.columns:
            col4.metric("True Anomalies", int(engine_labels.sum()))
        
        st.markdown("---")
        st.subheader("📊 Error Distribution (All Engines)")
        
        errors = autoencoder.compute_reconstruction_error(X_train_all)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors[y_train_all == 0], name='Normal',
            marker_color='green', opacity=0.6, nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=errors[y_train_all == 1], name='Anomaly',
            marker_color='red', opacity=0.6, nbinsx=50
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="blue",
                     annotation_text=f"Threshold: {threshold:.6f}")
        fig.update_layout(barmode='overlay', xaxis_title='Reconstruction Error',
                         yaxis_title='Frequency', height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        normal_errors = errors[y_train_all == 0]
        anomaly_errors = errors[y_train_all == 1]
        with col1:
            st.metric("Normal - Mean", f"{normal_errors.mean():.6f}")
            st.metric("Normal - Std", f"{normal_errors.std():.6f}")
        with col2:
            st.metric("Anomaly - Mean", f"{anomaly_errors.mean():.6f}")
            st.metric("Anomaly - Std", f"{anomaly_errors.std():.6f}")
    
    # LIVE MONITORING PAGE
    elif page == "📡 Live Monitoring":
        st.header("📡 Live Monitoring Simulation")
        st.info("ℹ️ Simulate real-time engine monitoring")
        
        engine_id = st.selectbox("Select Engine", sorted(train_df['engine_id'].unique()))
        
        with st.spinner("Preparing data..."):
            generator = SequenceGenerator(sequence_length=30, stride=1)
            train_df_scaled = scale_dataframe(scaler, train_df, feature_cols)
            X_all, y_all, meta_all = generator.generate_sequences(
                train_df_scaled, feature_cols, label_col='label'
            )
        
        engine_mask = meta_all['engine_ids'] == engine_id
        X_engine = X_all[engine_mask]
        y_engine = y_all[engine_mask]
        cycles_engine = meta_all['cycle_indices'][engine_mask]
        
        if len(X_engine) == 0:
            st.warning("⚠️ No data available")
            return
        
        st.success(f"✓ Loaded {len(X_engine)} cycles for Engine #{engine_id}")
        
        current_cycle_idx = st.slider("Current Cycle", 0, len(X_engine)-1, 0,
                                      help="Simulate different time points")
        
        current_sequence = X_engine[current_cycle_idx:current_cycle_idx+1]
        current_cycle = cycles_engine[current_cycle_idx]
        current_label = y_engine[current_cycle_idx]
        
        error = autoencoder.compute_reconstruction_error(current_sequence)[0]
        is_anomaly = error > threshold
        
        st.markdown("---")
        st.subheader("🎯 Current Status")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Current Cycle", current_cycle)
        col2.metric("Reconstruction Error", f"{error:.6f}")
        with col3:
            status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
            color = "red" if is_anomaly else "green"
            st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
        with col4:
            true_status = "🔴 Degraded" if current_label == 1 else "🟢 Healthy"
            st.markdown(f"<h3>{true_status}</h3>", unsafe_allow_html=True)
        
        progress = min((error / (threshold * 2)), 1.0)
        st.progress(progress, text=f"Error Level: {progress*100:.1f}%")
        
        percentage = (error / threshold) * 100
        st.metric("vs Threshold", f"{percentage:.1f}%")
        
        st.markdown("---")
        st.subheader("📈 Historical Trend")
        
        errors_so_far = autoencoder.compute_reconstruction_error(X_engine[:current_cycle_idx+1])
        cycles_so_far = cycles_engine[:current_cycle_idx+1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles_so_far, y=errors_so_far, mode='lines+markers',
            name='Error', line=dict(color='navy', width=2), marker=dict(size=4)
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold: {threshold:.6f}")
        fig.add_trace(go.Scatter(
            x=[current_cycle], y=[error], mode='markers', name='Current',
            marker=dict(color='yellow', size=15, symbol='star', 
                       line=dict(color='black', width=2))
        ))
        
        if 'label' in train_df.columns:
            engine_original = train_df[train_df['engine_id'] == engine_id]
            degradation = engine_original[engine_original['label'] == 1]
            if len(degradation) > 0:
                fig.add_vrect(
                    x0=degradation['cycle'].min(), x1=degradation['cycle'].max(),
                    fillcolor="orange", opacity=0.15, layer="below", line_width=0,
                    annotation_text="True Degradation", annotation_position="top left"
                )
        
        fig.update_layout(xaxis_title='Cycle', yaxis_title='Error',
                         height=450, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Error", f"{errors_so_far.min():.6f}")
        col2.metric("Max Error", f"{errors_so_far.max():.6f}")
        col3.metric("Mean Error", f"{errors_so_far.mean():.6f}")
        col4.metric("Anomalies", (errors_so_far > threshold).sum())
        
        with st.expander("🔍 Current Sensor Values"):
            st.write(f"**Cycle {current_cycle}:**")
            sensor_df = pd.DataFrame({
                'Sensor': feature_cols,
                'Value': current_sequence[0, -1, :]
            })
            st.dataframe(sensor_df)


if __name__ == "__main__":
    main()