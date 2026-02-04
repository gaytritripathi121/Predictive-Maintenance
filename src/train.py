import os
import sys
import pickle
import warnings
import joblib

warnings.filterwarnings('ignore')

from data_preprocessing import CMAPSSDataLoader
from sequence_generator import prepare_training_data
from lstm_autoencoder import LSTMAutoencoder
from anomaly_detection import AnomalyDetector
from visualization import Visualizer


class Config:
    DATA_PATH = 'data/'
    HEALTHY_RATIO = 0.35
    VARIANCE_THRESHOLD = 0.01

    SEQUENCE_LENGTH = 30
    STRIDE = 1
    VAL_RATIO = 0.2

    LATENT_DIM = 16
    EPOCHS = 100
    BATCH_SIZE = 64

    THRESHOLD_METHOD = 'mean_std'
    THRESHOLD_SIGMA = 3

    MODEL_DIR = 'models/'
    RESULTS_DIR = 'results/'


def ensure_directories():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)


def main():
    ensure_directories()

    print("="*70)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*70)
    
    loader = CMAPSSDataLoader(data_path=Config.DATA_PATH)

    train_df, test_df = loader.load_data()
    loader.perform_eda(save_plots=True)

    feature_cols, _ = loader.identify_informative_features(
        variance_threshold=Config.VARIANCE_THRESHOLD
    )

    train_df = loader.split_healthy_degradation(
        healthy_ratio=Config.HEALTHY_RATIO
    )

    train_df, test_df, scaler = loader.normalize_features(
        fit_on_healthy_only=True
    )

    # Save scaler
    scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
         joblib.dump(scaler, "scaler.joblib")
    print(f"✓ Scaler saved to {scaler_path}")

    print("\n" + "="*70)
    print("STEP 2: Preparing Training Sequences")
    print("="*70)
    
    data = prepare_training_data(
        train_df,
        test_df,
        feature_cols,
        sequence_length=Config.SEQUENCE_LENGTH,
        val_ratio=Config.VAL_RATIO
    )

    print("\n" + "="*70)
    print("STEP 3: Building and Training Model")
    print("="*70)
    
    autoencoder = LSTMAutoencoder(
        sequence_length=Config.SEQUENCE_LENGTH,
        n_features=data['n_features'],
        latent_dim=Config.LATENT_DIM
    )

    autoencoder.build_model()
    autoencoder.summary()
    
    autoencoder.train(
        data['X_train'],
        data['X_val'],
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )

    autoencoder.plot_training_history(
        save_path=os.path.join(Config.RESULTS_DIR, 'training_history.png')
    )

    print("\n" + "="*70)
    print("STEP 4: Saving Model (Multiple Formats for Compatibility)")
    print("="*70)
    
    # Save in multiple formats for maximum compatibility
    
    # 1. Save complete model (.keras format - primary)
    model_path = os.path.join(Config.MODEL_DIR, 'lstm_autoencoder.keras')
    autoencoder.save_model(model_path)
    
    # 2. Save weights only (.h5 format - fallback)
    weights_path = os.path.join(Config.MODEL_DIR, 'lstm_autoencoder.weights.h5')
    autoencoder.save_weights(weights_path)
    
    # 3. Save model configuration (for rebuilding)
    config_path = os.path.join(Config.MODEL_DIR, 'model_config.json')
    autoencoder.save_config(config_path)
    
    # 4. Save model parameters for easy rebuilding
    import json
    params_path = os.path.join(Config.MODEL_DIR, 'model_params.json')
    model_params = {
        'sequence_length': Config.SEQUENCE_LENGTH,
        'n_features': data['n_features'],
        'latent_dim': Config.LATENT_DIM
    }
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"✓ Model parameters saved to {params_path}")

    print("\n" + "="*70)
    print("STEP 5: Computing Anomaly Threshold")
    print("="*70)
    
    detector = AnomalyDetector(autoencoder)

    threshold = detector.compute_threshold(
        data['X_val'],
        method=Config.THRESHOLD_METHOD,
        sigma=Config.THRESHOLD_SIGMA
    )

    threshold_path = os.path.join(Config.MODEL_DIR, 'threshold.pkl')
    with open(threshold_path, 'wb') as f:
        pickle.dump(threshold, f)
    print(f"✓ Threshold saved to {threshold_path}")
    print(f"  Threshold value: {threshold:.6f}")

    print("\n" + "="*70)
    print("STEP 6: Evaluating Model Performance")
    print("="*70)
    
    metrics = detector.evaluate(
        data['X_train_eval'],
        data['y_train_eval']
    )

    detector.save_metrics(
        metrics,
        filepath=os.path.join(Config.RESULTS_DIR, 'metrics.json')
    )

    detector.plot_confusion_matrix(
        data['X_train_eval'],
        data['y_train_eval'],
        save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    )

    X_normal = data['X_train_eval'][data['y_train_eval'] == 0]
    X_anomaly = data['X_train_eval'][data['y_train_eval'] == 1]

    detector.plot_error_distribution(
        X_normal,
        X_anomaly,
        save_path=os.path.join(Config.RESULTS_DIR, 'error_distribution.png')
    )

    print("\n" + "="*70)
    print("STEP 7: Generating Visualizations")
    print("="*70)
    
    visualizer = Visualizer(autoencoder, detector)

    visualizer.plot_sensor_trends(
        train_df,
        engine_id=1,
        feature_cols=feature_cols,
        save_path=os.path.join(Config.RESULTS_DIR, 'sensor_trends.png')
    )

    visualizer.plot_reconstruction_error_timeline(
        train_df,
        data['X_train_eval'],
        data['meta_train_eval'],
        engine_id=None,
        save_path=os.path.join(
            Config.RESULTS_DIR,
            'reconstruction_error_timeline.png'
        )
    )

    visualizer.plot_detected_degradation_regions(
        train_df,
        data['X_train_eval'],
        data['meta_train_eval'],
        save_path=os.path.join(
            Config.RESULTS_DIR,
            'degradation_detection.png'
        )
    )

    visualizer.plot_reconstruction_comparison(
        data['X_train_eval'],
        idx=0,
        save_path=os.path.join(
            Config.RESULTS_DIR,
            'reconstruction_comparison.png'
        )
    )

    visualizer.generate_summary_report(
        metrics,
        save_path=os.path.join(
            Config.RESULTS_DIR,
            'summary_report.png'
        )
    )

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel files saved in: {Config.MODEL_DIR}")
    print(f"Results saved in: {Config.RESULTS_DIR}")
    print("\nSaved files:")
    print(f"  • lstm_autoencoder.keras (full model)")
    print(f"  • lstm_autoencoder.weights.h5 (weights only)")
    print(f"  • model_config.json (architecture)")
    print(f"  • model_params.json (hyperparameters)")
    print(f"  • scaler.pkl")
    print(f"  • threshold.pkl")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "="*70)
        print("ERROR OCCURRED!")
        print("="*70)
        traceback.print_exc()
        sys.exit(1)
