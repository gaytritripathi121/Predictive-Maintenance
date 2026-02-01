import os
import sys
import pickle
import warnings

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

    MODEL_DIR = '../models/'
    RESULTS_DIR = '../results/'


def ensure_directories():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)


def main():
    ensure_directories()

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

    with open(os.path.join(Config.MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    data = prepare_training_data(
        train_df,
        test_df,
        feature_cols,
        sequence_length=Config.SEQUENCE_LENGTH,
        val_ratio=Config.VAL_RATIO
    )

    autoencoder = LSTMAutoencoder(
        sequence_length=Config.SEQUENCE_LENGTH,
        n_features=data['n_features'],
        latent_dim=Config.LATENT_DIM
    )

    autoencoder.build_model()
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

    model_path = os.path.join(Config.MODEL_DIR, 'lstm_autoencoder.keras')
    autoencoder.save_model(model_path)

    detector = AnomalyDetector(autoencoder)

    threshold = detector.compute_threshold(
        data['X_val'],
        method=Config.THRESHOLD_METHOD,
        sigma=Config.THRESHOLD_SIGMA
    )

    with open(os.path.join(Config.MODEL_DIR, 'threshold.pkl'), 'wb') as f:
        pickle.dump(threshold, f)

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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
