# SmartFlush Predictive Model - Quick Start Guide

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/skalaliya/SmartFlush-PredictiveModel.git
cd SmartFlush-PredictiveModel
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Complete Pipeline

Execute the main script to run the entire analysis:

```bash
python main.py
```

This will:
- Load and combine data from Excel files (or create synthetic data if files not found)
- Perform exploratory data analysis
- Train multiple models (Ridge, LogReg, SVC, ANN)
- Optimize probability thresholds
- Calculate water savings and cost impact
- Generate plots and reports in `results/` and `reports/`

### Option 2: Use Jupyter Notebooks

#### Exploratory Data Analysis:
```bash
jupyter notebook notebooks/EDA.ipynb
```

#### ANN Development:
```bash
jupyter notebook notebooks/ANN.ipynb
```

### Option 3: Use as a Library

```python
from src.data_loading import load_and_combine_data, prepare_data
from src.models import train_ridge_model, train_ann_model
from src.metrics import evaluate_model
from src.utils import calculate_water_savings

# Load data
data_files = ['data/Combined_Data.xlsx', 'data/mon_fichier.xlsx']
df = load_and_combine_data(data_files)

# Prepare data
data_dict = prepare_data(df, target_col='flush_level', test_size=0.2)

# Train model
model, info = train_ridge_model(data_dict['X_train'], data_dict['y_train'])

# Evaluate
predictions = model.predict(data_dict['X_test'])
metrics = evaluate_model(data_dict['y_test'], predictions, 'Ridge')

# Calculate savings
savings = calculate_water_savings(data_dict['y_test'], predictions)
print(f"Water saved: {savings['savings']:.2f}L ({savings['savings_percentage']:.1f}%)")
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Data paths
data:
  combined_data: "data/Combined_Data.xlsx"
  additional_data: "data/mon_fichier.xlsx"

# Model hyperparameters
models:
  ridge:
    alphas: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  ann:
    hidden_layers: [128, 64, 32]
    dropout_rate: 0.2
    epochs: 100

# Hotel scenario
hotel:
  num_rooms: 100
  flushes_per_day: 5
  
# Water cost
water:
  cost_per_1000L: 4.0  # euros
```

## Running Tests

Execute the test suite:

```bash
# All tests
pytest

# Specific module
pytest tests/test_utils.py

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only (skip slow integration tests)
pytest -m "not slow and not integration"
```

## Project Structure

```
SmartFlush-PredictiveModel/
├── data/              # Excel data files go here
├── src/               # Source code modules
│   ├── data_loading.py
│   ├── eda.py
│   ├── models.py
│   ├── metrics.py
│   └── utils.py
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks
├── results/           # Output: plots, models
├── reports/           # Output: reports, tables
├── main.py            # Main pipeline script
└── config.yaml        # Configuration file
```

## Key Features

### Models Implemented:
1. **Ridge Regression** - with GridSearchCV and +1 adjustment
2. **Multinomial Logistic Regression** - with probability threshold optimization (MLR_2)
3. **SVC (Linear & Polynomial)** - with probability variants (SVC_2)
4. **Artificial Neural Network** - Keras with dropout, early stopping, learning curves

### Analysis Capabilities:
- VIF-based multicollinearity detection and removal
- Chi-squared tests for categorical features
- Pearson correlation analysis
- Comprehensive visualizations (heatmaps, pairplots, boxplots)
- Safe flush accuracy metric
- Water savings and cost impact calculations

### Flush Volume Mapping:
The system uses the following water volumes per flush level:

| Level | Volume (L) | Level | Volume (L) |
|-------|-----------|-------|-----------|
| 1     | 1.5       | 7     | 3.9       |
| 2     | 1.9       | 8     | 4.3       |
| 3     | 2.3       | 9     | 4.7       |
| 4     | 2.7       | 10    | 5.3       |
| 5     | 3.1       | 11    | 6.1       |
| 6     | 3.5       |       |           |

## Example Output

After running the pipeline, you'll find:

### In `results/`:
- `plots/eda/` - EDA visualizations
- `plots/correlation_heatmap.png` - Feature correlations
- `plots/model_comparison.png` - Model performance comparison
- `plots/confusion_matrix_*.png` - Confusion matrices
- `plots/ann_learning_curves.png` - ANN training history

### In `reports/`:
- `benchmark_comparison.csv` - All models vs competitor baseline
- `impact_report.json` - Water and cost savings analysis
- `classification_report_*.txt` - Detailed classification metrics
- `summary_statistics.csv` - Dataset statistics

## Troubleshooting

### No data files found
If you don't have data files, the pipeline will automatically create synthetic data for demonstration.

### Memory issues with large datasets
Adjust batch size in `config.yaml`:
```yaml
models:
  ann:
    batch_size: 16  # Reduce from 32
```

### Slow training
Use smaller parameter grids:
```yaml
models:
  ridge:
    alphas: [0.1, 1.0, 10.0]  # Instead of 6 values
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the comprehensive docstrings in each module
- Review the test files for usage examples

## Next Steps

1. Add your actual data files to `data/`
2. Update `config.yaml` with your parameters
3. Run `python main.py`
4. Review results in `results/` and `reports/`
5. Fine-tune models based on performance
6. Deploy the best model for production use

Happy modeling! 🚀💧
