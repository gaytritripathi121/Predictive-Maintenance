import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class Visualizer:
    def __init__(self, autoencoder, detector):
        self.autoencoder = autoencoder
        self.detector = detector

    def plot_sensor_trends(self, df, engine_id, feature_cols,
                           save_path='../results/sensor_trends.png'):
        engine_data = df[df['engine_id'] == engine_id]

        sensor_variance = engine_data[feature_cols].var().sort_values(ascending=False)
        top_sensors = sensor_variance.head(6).index.tolist()

        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f'Sensor Trends - Engine #{engine_id}', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for idx, sensor in enumerate(top_sensors):
            ax = axes[idx]
            ax.plot(engine_data['cycle'], engine_data[sensor], linewidth=1.5)

            if 'label' in engine_data.columns:
                degradation = engine_data[engine_data['label'] == 1]
                if len(degradation) > 0:
                    ax.axvspan(
                        degradation['cycle'].min(),
                        degradation['cycle'].max(),
                        alpha=0.2
                    )

            ax.set_xlabel('Cycle')
            ax.set_ylabel('Value')
            ax.set_title(sensor, fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_error_timeline(self, df, X_sequences, metadata,
                                           engine_id=None,
                                           save_path='../results/reconstruction_error_timeline.png'):
        errors = self.autoencoder.compute_reconstruction_error(X_sequences)

        error_df = pd.DataFrame({
            'engine_id': metadata['engine_ids'],
            'cycle': metadata['cycle_indices'],
            'reconstruction_error': errors
        })

        if engine_id is not None:
            engines_to_plot = [engine_id]
        else:
            engines_to_plot = error_df['engine_id'].unique()[:3]

        fig, axes = plt.subplots(len(engines_to_plot), 1,
                                 figsize=(14, 4 * len(engines_to_plot)))
        if len(engines_to_plot) == 1:
            axes = [axes]

        fig.suptitle('Reconstruction Error Over Engine Lifecycle',
                     fontsize=16, fontweight='bold')

        for ax, eng_id in zip(axes, engines_to_plot):
            engine_errors = error_df[error_df['engine_id'] == eng_id]
            engine_original = df[df['engine_id'] == eng_id]

            ax.plot(engine_errors['cycle'],
                    engine_errors['reconstruction_error'],
                    linewidth=2)

            ax.axhline(self.detector.threshold, linestyle='--', linewidth=2)

            anomalies = engine_errors[
                engine_errors['reconstruction_error'] > self.detector.threshold
            ]
            if len(anomalies) > 0:
                ax.scatter(anomalies['cycle'],
                           anomalies['reconstruction_error'],
                           s=50, zorder=5)

            if 'label' in engine_original.columns:
                degradation = engine_original[engine_original['label'] == 1]
                if len(degradation) > 0:
                    ax.axvspan(
                        degradation['cycle'].min(),
                        degradation['cycle'].max(),
                        alpha=0.15
                    )

            ax.set_xlabel('Cycle')
            ax.set_ylabel('Reconstruction Error')
            ax.set_title(f'Engine #{eng_id}', fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_detected_degradation_regions(self, df, X_sequences, metadata,
                                          save_path='../results/degradation_detection.png'):
        predictions, errors = self.detector.detect_anomalies(
            X_sequences, return_errors=True
        )

        results_df = pd.DataFrame({
            'engine_id': metadata['engine_ids'],
            'cycle': metadata['cycle_indices'],
            'prediction': predictions,
            'error': errors
        })

        first_detections = []
        for eng_id in results_df['engine_id'].unique():
            engine_pred = results_df[results_df['engine_id'] == eng_id]
            anomalies = engine_pred[engine_pred['prediction'] == 1]
            if len(anomalies) > 0:
                first_cycle = anomalies['cycle'].min()
                max_cycle = engine_pred['cycle'].max()
                first_detections.append({
                    'engine_id': eng_id,
                    'first_detection': first_cycle,
                    'max_cycle': max_cycle
                })

        detection_df = pd.DataFrame(first_detections)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Degradation Detection Analysis',
                     fontsize=16, fontweight='bold')

        if len(detection_df) > 0:
            axes[0].hist(detection_df['first_detection'], bins=20, alpha=0.7)
            axes[0].axvline(detection_df['first_detection'].median(),
                            linestyle='--', linewidth=2)
            axes[0].set_xlabel('First Detection Cycle')
            axes[0].set_ylabel('Engine Count')
            axes[0].grid(alpha=0.3)

        sample_engines = results_df['engine_id'].unique()[:15]
        for eng_id in sample_engines:
            engine_data = results_df[results_df['engine_id'] == eng_id]
            normal = engine_data[engine_data['prediction'] == 0]
            anomaly = engine_data[engine_data['prediction'] == 1]

            axes[1].scatter(normal['cycle'], [eng_id] * len(normal), s=20, alpha=0.5)
            axes[1].scatter(anomaly['cycle'], [eng_id] * len(anomaly), s=20, alpha=0.8)

        axes[1].set_xlabel('Cycle')
        axes[1].set_ylabel('Engine ID')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_comparison(self, X_sample, idx=0,
                                       save_path='../results/reconstruction_comparison.png'):
        original = X_sample[idx]
        reconstructed = self.autoencoder.model.predict(
            X_sample[idx:idx + 1], verbose=0
        )[0]

        n_features = min(6, original.shape[1])

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Original vs Reconstructed',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for feat_idx in range(n_features):
            ax = axes[feat_idx]
            t = np.arange(len(original))
            ax.plot(t, original[:, feat_idx], linewidth=2)
            ax.plot(t, reconstructed[:, feat_idx],
                    linestyle='--', linewidth=2)
            ax.set_title(f'Feature {feat_idx + 1}', fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, metrics,
                                save_path='../results/summary_report.png'):
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('Performance Summary',
                     fontsize=18, fontweight='bold')

        metric_items = [
            ('Precision', metrics['precision']),
            ('Recall', metrics['recall']),
            ('F1-Score', metrics['f1_score']),
            ('Accuracy', metrics['accuracy'])
        ]

        for i, (name, value) in enumerate(metric_items):
            ax = fig.add_subplot(gs[0, i] if i < 3 else gs[1, i - 3])
            ax.text(0.5, 0.6, f'{value:.3f}',
                    fontsize=42, ha='center', va='center', fontweight='bold')
            ax.text(0.5, 0.2, name, fontsize=16,
                    ha='center', va='center')
            ax.axis('off')

        ax_cm = fig.add_subplot(gs[:, 2])
        cm = [
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ]
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix', fontweight='bold')

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
