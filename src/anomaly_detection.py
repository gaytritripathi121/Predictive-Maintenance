import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json


class AnomalyDetector:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.threshold = None
        self.validation_errors = None

    def compute_threshold(self, X_val, method='mean_std', sigma=3):
        self.validation_errors = self.autoencoder.compute_reconstruction_error(X_val)

        if method == 'mean_std':
            mean_error = np.mean(self.validation_errors)
            std_error = np.std(self.validation_errors)
            self.threshold = mean_error + sigma * std_error

        elif method == 'percentile':
            self.threshold = np.percentile(self.validation_errors, sigma)

        return self.threshold

    def detect_anomalies(self, X, return_errors=False):
        if self.threshold is None:
            raise ValueError("Threshold not set")

        errors = self.autoencoder.compute_reconstruction_error(X)
        predictions = (errors > self.threshold).astype(int)

        if return_errors:
            return predictions, errors
        return predictions

    def evaluate(self, X, y_true, name='Dataset'):
        y_pred, errors = self.detect_anomalies(X, return_errors=True)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold': float(self.threshold),
            'mean_error_normal': float(errors[y_true == 0].mean()) if (y_true == 0).any() else 0,
            'mean_error_anomaly': float(errors[y_true == 1].mean()) if (y_true == 1).any() else 0
        }

        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

        return metrics

    def plot_confusion_matrix(self, X, y_true, save_path='../results/confusion_matrix.png'):
        y_pred = self.detect_anomalies(X)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_error_distribution(self, X_normal, X_anomaly, save_path='../results/error_distribution.png'):
        errors_normal = self.autoencoder.compute_reconstruction_error(X_normal)
        errors_anomaly = self.autoencoder.compute_reconstruction_error(X_anomaly)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(errors_normal, bins=50, alpha=0.6, label='Normal')
        ax.hist(errors_anomaly, bins=50, alpha=0.6, label='Anomaly')

        ax.axvline(self.threshold, linestyle='--', linewidth=2, label='Threshold')

        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, metrics, filepath='../results/metrics.json'):
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
