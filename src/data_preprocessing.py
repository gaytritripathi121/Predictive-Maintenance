import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CMAPSSDataLoader:

    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        self.scaler = None

        self.index_cols = ['engine_id', 'cycle']
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_cols + self.setting_cols + self.sensor_cols

    def load_data(self):
        self.train_df = pd.read_csv(
            f'{self.data_path}train_FD001.txt',
            sep=r'\s+',
            header=None,
            names=self.col_names
        )

        self.test_df = pd.read_csv(
            f'{self.data_path}test_FD001.txt',
            sep=r'\s+',
            header=None,
            names=self.col_names
        )

        return self.train_df, self.test_df

    def perform_eda(self, save_plots=True):
        engine_cycles = self.train_df.groupby('engine_id')['cycle'].max()

        if save_plots:
            self._plot_eda(engine_cycles)

    def _plot_eda(self, engine_cycles):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].hist(engine_cycles, bins=30, alpha=0.7)
        axes[0, 0].axvline(engine_cycles.mean(), linestyle='--')
        axes[0, 0].set_xlabel('Cycles to Failure')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Engine Lifecycles')

        sensor_variance = self.train_df[self.sensor_cols].var().sort_values(ascending=False)
        axes[0, 1].bar(range(len(sensor_variance)), sensor_variance.values, alpha=0.7)
        axes[0, 1].set_xlabel('Sensor Index')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title('Sensor Variance')

        sample_engine = self.train_df[self.train_df['engine_id'] == 1]
        for sensor in ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_15']:
            axes[1, 0].plot(sample_engine['cycle'], sample_engine[sensor], alpha=0.7)
        axes[1, 0].set_xlabel('Cycle')
        axes[1, 0].set_ylabel('Sensor Value')
        axes[1, 0].set_title('Sensor Trends')

        axes[1, 1].scatter(
            self.train_df['setting_1'],
            self.train_df['setting_2'],
            alpha=0.3,
            s=1
        )
        axes[1, 1].set_xlabel('Setting 1')
        axes[1, 1].set_ylabel('Setting 2')
        axes[1, 1].set_title('Operating Conditions')

        plt.tight_layout()
        plt.savefig('../results/eda_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def identify_informative_features(self, variance_threshold=0.01):
        sensor_variance = self.train_df[self.sensor_cols].var()
        low_variance_sensors = sensor_variance[sensor_variance < variance_threshold].index.tolist()

        columns_to_drop = self.setting_cols + low_variance_sensors
        self.feature_cols = [c for c in self.sensor_cols if c not in low_variance_sensors]

        return self.feature_cols, columns_to_drop

    def split_healthy_degradation(self, healthy_ratio=0.35):
        self.train_df['RUL'] = 0
        self.train_df['label'] = 0

        for engine_id in self.train_df['engine_id'].unique():
            mask = self.train_df['engine_id'] == engine_id
            engine_data = self.train_df[mask]
            max_cycle = engine_data['cycle'].max()

            self.train_df.loc[mask, 'RUL'] = max_cycle - engine_data['cycle']
            healthy_threshold = int(max_cycle * healthy_ratio)

            self.train_df.loc[
                mask & (self.train_df['cycle'] > healthy_threshold),
                'label'
            ] = 1

        return self.train_df

    def normalize_features(self, fit_on_healthy_only=True):
        self.scaler = StandardScaler()

        if fit_on_healthy_only:
            healthy_data = self.train_df[self.train_df['label'] == 0][self.feature_cols]
            self.scaler.fit(healthy_data)
        else:
            self.scaler.fit(self.train_df[self.feature_cols])

        self.train_df[self.feature_cols] = self.scaler.transform(
            self.train_df[self.feature_cols]
        )
        self.test_df[self.feature_cols] = self.scaler.transform(
            self.test_df[self.feature_cols]
        )

        return self.train_df, self.test_df, self.scaler
