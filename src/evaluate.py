import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

from data_preprocessing import CMAPSSDataLoader
from sequence_generator import SequenceGenerator
from lstm_autoencoder import LSTMAutoencoder
from anomaly_detection import AnomalyDetector
from visualization import Visualizer


def load_rul_labels(filepath='../data/RUL_FD001.txt'):
    try:
        rul_values = pd.read_csv(filepath, header=None, names=['RUL'])
        return {i + 1: rul for i, rul in enumerate(rul_values['RUL'])}
    except FileNotFoundError:
        return None


def add_test_labels(test_df, rul_dict, degradation_threshold=50):
    if rul_dict is None:
        return test_df

    test_df = test_df.copy()
    test_df['label'] = 0

    for engine_id in test_df['engine_id'].unique():
        mask = test_df['engine_id'] == engine_id
        engine_data = test_df[mask]

        max_cycle = engine_data['cycle'].max()
        rul_at_end = rul_dict.get(engine_id, 0)
        total_lifecycle = max_cycle + rul_at_end

        test_df.loc[mask, 'RUL'] = total_lifecycle - engine_data['cycle']
        test_df.loc[
            mask & (test_df['RUL'] <= degradation_threshold),
            'label'
        ] = 1

    return test_df


def main():
    scaler_path = '../models/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model_path_keras = '../models/lstm_autoencoder.keras'
    model_path_h5 = '../models/lstm_autoencoder.h5'

    if os.path.exists(model_path_keras):
        model = keras.models.load_model(model_path_keras)
    elif os.path.exists(model_path_h5):
        model = keras.models.load_model(model_path_h5, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        raise FileNotFoundError('Model not found')

    threshold_path = '../models/threshold.pkl'
    with open(threshold_path, 'rb') as f:
        threshold = pickle.load(f)

    loader = CMAPSSDataLoader(data_path='../data/')
    _, test_df = loader.load_data()

    feature_cols, _ = loader.identify_informative_features()

    rul_dict = load_rul_labels('../data/RUL_FD001.txt')
    if rul_dict is not None:
        test_df = add_test_labels(test_df, rul_dict)

    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    generator = SequenceGenerator(sequence_length=30, stride=1)

    if rul_dict is not None:
        X_test, y_test, meta_test = generator.generate_sequences(
            test_df,
            feature_cols,
            label_col='label'
        )
    else:
        X_test, meta_test = generator.generate_sequences(
            test_df,
            feature_cols,
            label_col=None
        )
        y_test = None

    autoencoder = LSTMAutoencoder(
        sequence_length=30,
        n_features=len(feature_cols)
    )
    autoencoder.model = model

    detector = AnomalyDetector(autoencoder)
    detector.threshold = threshold

    if y_test is not None:
        metrics = detector.evaluate(X_test, y_test, name='Test Set')

        detector.save_metrics(
            metrics,
            filepath='../results/test_metrics.json'
        )

        detector.plot_confusion_matrix(
            X_test, y_test,
            save_path='../results/test_confusion_matrix.png'
        )

        X_normal = X_test[y_test == 0]
        X_anomaly = X_test[y_test == 1]

        detector.plot_error_distribution(
            X_normal,
            X_anomaly,
            save_path='../results/test_error_distribution.png'
        )
    else:
        detector.detect_anomalies(X_test)

    visualizer = Visualizer(autoencoder, detector)

    visualizer.plot_reconstruction_error_timeline(
        test_df,
        X_test,
        meta_test,
        engine_id=None,
        save_path='../results/test_reconstruction_timeline.png'
    )

    visualizer.plot_sensor_trends(
        test_df,
        engine_id=1,
        feature_cols=feature_cols,
        save_path='../results/test_sensor_trends.png'
    )

    if y_test is not None:
        visualizer.plot_detected_degradation_regions(
            test_df,
            X_test,
            meta_test,
            save_path='../results/test_degradation_detection.png'
        )

        visualizer.generate_summary_report(
            metrics,
            save_path='../results/test_summary_report.png'
        )


if __name__ == '__main__':
    main()
