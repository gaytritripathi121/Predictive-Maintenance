import os
import json
import logging
from datetime import datetime


def setup_logging(log_dir='../logs', log_filename=None):
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'training_{timestamp}.log'

    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_config(config_dict, filepath='../results/config.json'):
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


def load_config(filepath='../results/config.json'):
    with open(filepath, 'r') as f:
        return json.load(f)


def create_project_structure():
    directories = [
        '../data',
        '../models',
        '../results',
        '../logs',
        '../notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'w').close()


def print_banner(text, width=70):
    print("\n" + "=" * width)
    padding = (width - len(text) - 2) // 2
    print(" " * padding + text)
    print("=" * width + "\n")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_gpu_memory_info():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return {
                'available': True,
                'count': len(gpus),
                'devices': [gpu.name for gpu in gpus]
            }
        return {'available': False}
    except Exception:
        return {'available': False}


def set_seed(seed=42):
    import numpy as np
    import tensorflow as tf
    import random

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


class MetricsTracker:
    def __init__(self, filepath='../results/metrics_history.json'):
        self.filepath = filepath
        self.metrics = {}
        if os.path.exists(filepath):
            self.load()

    def add(self, key, value):
        self.metrics.setdefault(key, []).append(value)

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def load(self):
        with open(self.filepath, 'r') as f:
            self.metrics = json.load(f)

    def get_latest(self, key):
        values = self.metrics.get(key)
        return values[-1] if values else None

    def get_all(self, key):
        return self.metrics.get(key, [])


def validate_data_files(data_path='../data/'):
    required_files = ['train_FD001.txt', 'test_FD001.txt']

    missing = [
        f for f in required_files
        if not os.path.exists(os.path.join(data_path, f))
    ]

    if missing:
        return False

    return True
