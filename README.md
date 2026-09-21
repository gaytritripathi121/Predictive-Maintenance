# Predictive Maintenance using LSTM Autoencoder

## Overview

This project implements a **predictive maintenance system** for detecting abnormal engine behavior and identifying potential degradation before failure.

The system uses **time-series sensor data from the NASA CMAPSS dataset** and an **LSTM Autoencoder** to learn patterns from normal operating conditions. Anomalies are identified using reconstruction error.

### Key Features

* LSTM Autoencoder for temporal pattern recognition
* Unsupervised anomaly detection
* 80.70% recall on the test data
* 53.24% precision on the test data
* Interactive Streamlit dashboard
* Modular ML pipeline with preprocessing, training, evaluation, and visualization

---

## Problem Statement

### Challenge

Unexpected turbofan engine failures can result in:

* Significant maintenance and downtime costs
* Operational disruptions
* Safety concerns
* Reduced equipment availability

### Solution

This project analyzes sensor measurements collected over multiple operating cycles to identify patterns associated with abnormal engine behavior.

The system learns the characteristics of **normal engine operation** and flags sequences with unusually high reconstruction error as potential anomalies.

> **Note:** The model's anomaly detection results are based on the NASA CMAPSS dataset and should not be interpreted as a production-ready aircraft maintenance system without further validation.

---

## Results

| Metric        | Value  | Meaning                                                       |
| ------------- | ------ | ------------------------------------------------------------- |
| **Recall**    | 80.70% | Detects approximately 81% of the labeled anomalies            |
| **Precision** | 53.24% | Approximately 53% of detected anomalies were actual anomalies |
| **F1-Score**  | 64.15% | Balance between precision and recall                          |
| **Accuracy**  | 92.16% | Overall classification accuracy                               |

### Potential Business Impact

Predictive maintenance can potentially help organizations:

* Reduce unexpected equipment failures
* Minimize unplanned downtime
* Improve maintenance planning
* Extend equipment availability
* Reduce unnecessary maintenance

The actual financial impact would depend on the deployment environment, maintenance strategy, equipment type, and operational costs.

---

## Architecture

```text
Input: 30 timesteps × 11 sensors
              ↓
       Encoder
   LSTM(64) → LSTM(32)
              ↓
        Dense(16)
              ↓
       Latent Space
       16 features
              ↓
        Decoder
   LSTM(32) → LSTM(64)
              ↓
        Dense(11)
              ↓
   Reconstructed Sequence
              ↓
     Reconstruction Error
              ↓
   Error > Threshold
              ↓
      Anomaly Detected
```

### Why LSTM Autoencoder?

**LSTM:** Captures temporal dependencies and changes in sensor behavior over time.

**Autoencoder:** Learns to reconstruct normal operating patterns without requiring failure labels during training.

**Anomaly Detection:** When the model encounters an unusual sequence, its reconstruction error increases. A threshold is then used to identify potential anomalies.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gaytritripathi121/Predictive-Maintenance.git
cd Predictive-Maintenance
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the **NASA CMAPSS FD001** dataset from the NASA Prognostics Data Repository.

Place the required dataset files inside the `data/` folder:

```text
data/
├── train_FD001.txt
└── test_FD001.txt
```

### 4. Train the Model

```bash
cd src
python train.py
```

Training time may vary depending on the hardware and configuration.

### 5. Evaluate the Model

```bash
python evaluate.py
```

### 6. Launch the Dashboard

From the project root:

```bash
streamlit run streamlit_app/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Project Structure

```text
Predictive-Maintenance/
├── data/                       # Dataset files
├── src/                        # Source code
│   ├── data_preprocessing.py
│   ├── sequence_generator.py
│   ├── lstm_autoencoder.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   ├── train.py                # Model training pipeline
│   └── evaluate.py             # Model evaluation
├── models/                     # Saved model and preprocessing artifacts
├── results/                    # Evaluation results and visualizations
├── streamlit_app/              # Streamlit dashboard
└── requirements.txt
```

---

## Key Technical Decisions

### 1. Preventing Data Leakage

The scaler is fitted only on the training data used to represent healthy operating conditions.

This helps ensure that information from the evaluation data does not influence preprocessing during training.

### 2. Using Precision, Recall and F1-Score

Because anomaly detection can involve imbalanced classes, accuracy alone may not provide a complete picture of model performance.

Therefore, Precision, Recall, and F1-Score are also used for evaluation.

### 3. Anomaly Threshold

The anomaly threshold is calculated using the reconstruction errors from the validation data.

A threshold based on the mean reconstruction error plus three standard deviations is used as a conservative starting point.

### 4. Sequence Generation

The model processes data using **30-timestep sliding windows**.

This allows the LSTM to learn temporal relationships between sensor measurements across multiple operating cycles.

---

## Visualizations

The project generates visualizations including:

| Visualization             | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| **Sensor Trends**         | Shows sensor behavior across engine operating cycles                |
| **Reconstruction Error**  | Displays reconstruction error and detected anomalies                |
| **Confusion Matrix**      | Shows classification performance                                    |
| **Error Distribution**    | Compares reconstruction errors between normal and anomalous samples |
| **Degradation Detection** | Helps visualize changes in engine behavior over time                |

---

## Technologies Used

* **Python 3.8+**
* **TensorFlow / Keras** – Deep learning and LSTM Autoencoder
* **Pandas** – Data processing
* **NumPy** – Numerical computation
* **Scikit-learn** – Preprocessing and evaluation
* **Matplotlib / Seaborn** – Data visualization
* **Streamlit** – Interactive dashboard

---

## Dataset

### NASA CMAPSS

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset.

For the FD001 subset:

* 100 training engine trajectories
* 100 test engine trajectories
* 21 sensor measurements
* Single operating condition
* Single fault mode

The model uses a selected subset of sensor features for training.

---

## What I Learned

Through this project, I gained practical experience in:

* Time-series anomaly detection using LSTMs
* Unsupervised learning
* Autoencoder-based anomaly detection
* Data preprocessing and feature selection
* Preventing data leakage in ML pipelines
* Evaluating models using Precision, Recall, and F1-Score
* Working with imbalanced anomaly-detection datasets
* Building an end-to-end machine learning pipeline
* Deploying ML models through Streamlit

---

## Future Improvements

1. **Multi-condition training** using FD002–FD004 datasets
2. **Attention mechanisms** to identify important sensor contributions
3. **Transfer learning** across different engine conditions
4. **Remaining Useful Life (RUL) prediction**
5. **Real-time monitoring** using streaming technologies such as Kafka
6. **Model serving** through a dedicated API
7. **Hyperparameter optimization** for improved anomaly detection performance

---

## Author

**Gaytri Tripathi**

* GitHub: https://github.com/gaytritripathi121
* Email: [gaytritripathi121@gmail.com](mailto:gaytritripathi121@gmail.com)

---

## Acknowledgments

* NASA Ames Research Center for providing the CMAPSS dataset

---

## Keywords

**Predictive Maintenance, LSTM, Autoencoder, Anomaly Detection, Deep Learning, Time Series, TensorFlow, Keras, Turbofan, NASA CMAPSS, Machine Learning, Python**



