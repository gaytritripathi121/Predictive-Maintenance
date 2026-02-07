##  Overview

A production-ready ML pipeline that automatically evaluates dataset quality and generates a 0-100 health score with actionable recommendations. Solves the critical problem that **80% of ML failures come from data quality issues, not models**.

##  Key Features

- **Automatic Quality Assessment** - Missing values, duplicates, class imbalance, outliers
- **ML-Based Anomaly Detection** - Isolation Forest ensemble for intelligent pattern detection
- **Smart Scoring System** - 0-100 health score across 4 dimensions (Completeness, Consistency, Balance, Integrity)
- **Visual Reports** - 6+ auto-generated charts with actionable insights
- **Multiple Interfaces** - Python API, CLI, and Interactive Web Dashboard

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dataset-quality-auditor.git
cd dataset-quality-auditor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Option 1: Web Dashboard** (Recommended)
```bash
streamlit run app.py
```

**Option 2: Command Line**
```bash
python cli.py audit --file data/your_dataset.csv
```

**Option 3: Python API**
```python
from main import DatasetQualityAuditor

auditor = DatasetQualityAuditor()
results = auditor.audit_dataset('data/your_data.csv')
print(f"Health Score: {results['scores']['overall_score']}/100")
```

##  Example Output

```
Dataset Health Score: 91.25/100 (Excellent)

Component Scores:
├─ Completeness: 25.0/25 (100%)
├─ Consistency: 25.0/25 (100%)
├─ Balance: 18.75/25 (75%)
└─ Integrity: 22.5/25 (90%)

Key Findings:
✓ No missing values
✓ No duplicates
⚠ Mild class imbalance (6:1 ratio) - normal for churn datasets
⚠ 92 statistical outliers detected - review recommended
```

##  Tech Stack

- **Python 3.8+** - Core language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning (Isolation Forest)
- **Matplotlib & Seaborn** - Visualizations
- **Streamlit** - Web interface

##  Project Structure

```
dataset-quality-auditor/
├── src/                    # Core modules
│   ├── dataset_loader.py   # Data loading & type detection
│   ├── data_profiler.py    # Statistical profiling
│   ├── quality_checker.py  # Rule-based validation
│   ├── ml_anomaly_detector.py  # ML models
│   ├── scoring_engine.py   # Health score calculation
│   └── report_generator.py # Report creation
├── models/                 # ML model implementations
├── config/                 # Configuration files
├── main.py                 # Main pipeline
├── cli.py                  # Command-line interface
└── app.py                  # Streamlit dashboard
```

##  How It Works

1. **Load Dataset** - Supports CSV, Excel, Parquet
2. **Profile Data** - Statistical analysis and metadata extraction
3. **Quality Checks** - Rule-based validation (missing values, duplicates, etc.)
4. **ML Detection** - Isolation Forest identifies statistical anomalies
5. **Score Calculation** - Weighted scoring across 4 dimensions
6. **Generate Reports** - Visual dashboards + JSON/HTML/Text reports

##  Scoring Methodology

```
Overall Score = 0.25 × Completeness + 
                0.25 × Consistency + 
                0.25 × Balance + 
                0.25 × Integrity
```

- **Completeness** (0-100): Missing value analysis
- **Consistency** (0-100): Duplicate detection
- **Balance** (0-100): Class distribution (domain-calibrated thresholds)
- **Integrity** (0-100): Anomaly detection via ML

##  Use Cases

- **Pre-Training Validation** - Catch data issues before wasting compute
- **Data Pipeline Monitoring** - Continuous quality assessment
- **Exploratory Analysis** - Quick dataset health check
- **ML Pipeline Integration** - Automated quality gates


##  Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

##  Acknowledgments

Built with Python, Scikit-learn, and Streamlit. Inspired by the need for automated data quality assessment in production ML workflows.

---

