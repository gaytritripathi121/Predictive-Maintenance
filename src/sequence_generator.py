import numpy as np
import pandas as pd


class SequenceGenerator:

    def __init__(self, sequence_length=30, stride=1):
        self.sequence_length = sequence_length
        self.stride = stride

    def generate_sequences(self, df, feature_cols, label_col=None, engine_id_col='engine_id'):
        sequences = []
        labels = []
        engine_ids = []
        cycle_indices = []

        for engine_id in df[engine_id_col].unique():
            engine_data = df[df[engine_id_col] == engine_id].sort_values('cycle')
            features = engine_data[feature_cols].values

            for i in range(0, len(features) - self.sequence_length + 1, self.stride):
                sequences.append(features[i:i + self.sequence_length])
                engine_ids.append(engine_id)
                cycle_indices.append(i + self.sequence_length - 1)

                if label_col is not None:
                    labels.append(
                        engine_data.iloc[i + self.sequence_length - 1][label_col]
                    )

        X = np.array(sequences)

        metadata = {
            'engine_ids': np.array(engine_ids),
            'cycle_indices': np.array(cycle_indices),
            'n_sequences': len(sequences),
            'sequence_length': self.sequence_length,
            'n_features': len(feature_cols)
        }

        if label_col is not None:
            return X, np.array(labels), metadata

        return X, metadata

    def split_train_val(self, X, y, metadata, val_ratio=0.2, random_state=42):
        np.random.seed(random_state)

        unique_engines = np.unique(metadata['engine_ids'])
        n_val_engines = int(len(unique_engines) * val_ratio)

        val_engines = np.random.choice(
            unique_engines, n_val_engines, replace=False
        )

        val_mask = np.isin(metadata['engine_ids'], val_engines)
        train_mask = ~val_mask

        return (
            X[train_mask],
            X[val_mask],
            y[train_mask],
            y[val_mask]
        )

    def filter_by_label(self, X, y, metadata, target_label=0):
        mask = y == target_label

        filtered_metadata = {
            'engine_ids': metadata['engine_ids'][mask],
            'cycle_indices': metadata['cycle_indices'][mask],
            'n_sequences': mask.sum(),
            'sequence_length': metadata['sequence_length'],
            'n_features': metadata['n_features']
        }

        return X[mask], y[mask], filtered_metadata


def prepare_training_data(
    train_df,
    test_df,
    feature_cols,
    sequence_length=30,
    val_ratio=0.2
):
    generator = SequenceGenerator(
        sequence_length=sequence_length,
        stride=1
    )

    X_train_all, y_train_all, meta_train = generator.generate_sequences(
        train_df,
        feature_cols,
        label_col='label'
    )

    X_healthy, y_healthy, meta_healthy = generator.filter_by_label(
        X_train_all,
        y_train_all,
        meta_train,
        target_label=0
    )

    X_train, X_val, _, _ = generator.split_train_val(
        X_healthy,
        y_healthy,
        meta_healthy,
        val_ratio=val_ratio
    )

    X_test, meta_test = generator.generate_sequences(
        test_df,
        feature_cols,
        label_col=None
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_train_eval': X_train_all,
        'y_train_eval': y_train_all,
        'meta_train_eval': meta_train,
        'X_test': X_test,
        'meta_test': meta_test,
        'sequence_length': sequence_length,
        'n_features': X_train.shape[2]
    }
