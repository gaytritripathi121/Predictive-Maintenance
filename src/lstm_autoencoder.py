import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


class LSTMAutoencoder:

    def __init__(self, sequence_length, n_features, latent_dim=16):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = None
        self.history = None

    def build_model(self):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))

        encoded = layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
        encoded = layers.Dropout(0.2)(encoded)

        encoded = layers.LSTM(32, activation='tanh', return_sequences=False)(encoded)
        encoded = layers.Dropout(0.2)(encoded)

        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(encoded)

        decoded = layers.RepeatVector(self.sequence_length)(latent)

        decoded = layers.LSTM(32, activation='tanh', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)

        decoded = layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)

        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features)
        )(decoded)

        self.model = Model(
            inputs=inputs,
            outputs=outputs,
            name='LSTM_Autoencoder'
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def summary(self):
        if self.model is not None:
            self.model.summary()

    def train(self, X_train, X_val, epochs=100, batch_size=64, verbose=1):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ModelCheckpoint(
                '../models/lstm_autoencoder_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]

        self.history = self.model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def plot_training_history(self, save_path='../results/training_history.png'):
        if self.history is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(self.history.history['loss'], linewidth=2, label='Training Loss')
        ax.plot(self.history.history['val_loss'], linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compute_reconstruction_error(self, X):
        X_reconstructed = self.model.predict(X, verbose=0)

        return np.mean(
            np.square(X - X_reconstructed),
            axis=(1, 2)
        )

    def save_model(self, filepath='../models/lstm_autoencoder.keras'):
        """Save model in .keras format"""
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')

        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")

    def save_weights(self, filepath='../models/lstm_autoencoder.weights.h5'):
        """Save only model weights (more compatible)"""
        self.model.save_weights(filepath)
        print(f"✓ Weights saved to {filepath}")

    def save_config(self, filepath='../models/model_config.json'):
        """Save model configuration"""
        import json
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'model_config': self.model.get_config()
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Config saved to {filepath}")

    def load_model(self, filepath='../models/lstm_autoencoder.keras'):
        """Load complete model"""
        self.model = keras.models.load_model(filepath, compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model

    def load_weights(self, filepath='../models/lstm_autoencoder.weights.h5'):
        """Load only weights (requires model to be built first)"""
        if self.model is None:
            raise ValueError("Model must be built before loading weights. Call build_model() first.")
        self.model.load_weights(filepath)
        return self.model
