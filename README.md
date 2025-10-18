# SmartFlush-PredictiveModel

A Python-based predictive model for optimizing flush volume based on sensor data, aimed at water conservation through intelligent prediction.

## Overview

SmartFlush-PredictiveModel uses machine learning to predict optimal flush volumes based on sensor inputs, helping to minimize water waste while ensuring effective flushing. The system implements multiple ML models including Ridge Regression, Multiple Linear Regression (MLR), Support Vector Regression (SVR), and Artificial Neural Networks (ANN) using Keras.

## Features

- **Data Loading**: Excel file support with comprehensive data cleaning
- **Exploratory Data Analysis (EDA)**: 
  - Chi-square tests for categorical features
  - Pearson correlation analysis
  - Automated visualization (heatmaps, distributions, pairplots)
- **Feature Engineering**:
  - Variance Inflation Factor (VIF) analysis for multicollinearity detection
  - Polynomial feature generation
  - Multiple scaling methods (Standard, MinMax, Robust)
- **Machine Learning Models**:
  - Ridge Regression with GridSearchCV
  - Multiple Linear Regression (MLR)
  - Support Vector Regression (SVR)
  - Artificial Neural Network (ANN/Keras)
- **Custom Metrics**:
  - Safe flush accuracy
  - Water savings calculations
  - Standard regression metrics (MAE, RMSE, R²)
- **Comprehensive Logging**: Detailed execution logs for debugging and analysis
- **Testing**: Full pytest test suite

## Project Structure

```
SmartFlush-PredictiveModel/
├── data/                    # Data directory for Excel files
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loading.py     # Excel data loading and preprocessing
│   ├── eda.py             # Exploratory data analysis
│   ├── models.py          # ML models (Ridge, MLR, SVC, ANN)
│   ├── metrics.py         # Custom metrics (safe_flush_accuracy, water savings)
│   └── utils.py           # Utilities (VIF, scaling, polynomial features)
├── reports/               # Generated reports and figures
│   └── figures/          # EDA plots and visualizations
├── results/              # Model results and saved models
│   └── models/          # Serialized trained models
├── tests/               # Pytest test suite
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_eda.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_utils.py
├── main.py             # Main entry point
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/skalaliya/SmartFlush-PredictiveModel.git
cd SmartFlush-PredictiveModel
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Place your sensor data Excel file in the `data/` directory with the following structure:
   - Feature columns: sensor readings (e.g., flow_rate, pressure, temperature)
   - Target column: `flush_volume` (or specify custom name)

2. Run the complete pipeline:
```bash
python main.py
```

### Custom Configuration

Edit `config.yaml` to customize:
- Data paths and parameters
- Feature engineering settings (VIF threshold, polynomial degree, scaling method)
- Model hyperparameters
- Grid search settings
- Logging configuration

### Command Line Options

```bash
# Use custom config file
python main.py --config custom_config.yaml

# Specify data file
python main.py --data path/to/data.xlsx

# Specify custom target column
python main.py --target custom_target_name
```

### Output

The pipeline generates:
- **EDA Reports**: `reports/figures/` - correlation heatmaps, distributions, etc.
- **Trained Models**: `results/models/` - serialized models
- **Metrics Comparison**: `results/model_comparison.csv` - performance comparison
- **Logs**: `logs/smartflush.log` - detailed execution logs

## Metrics

### Safe Flush Accuracy
Measures the percentage of predictions that meet the safety threshold (default: predicted ≥ 95% of actual volume).

### Water Savings
Calculates water savings compared to baseline flush volume:
- Total savings (liters)
- Efficiency percentage
- Average waste/shortage per flush

### Standard Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Models

### Ridge Regression
L2-regularized linear regression with hyperparameter tuning via GridSearchCV.

### Multiple Linear Regression (MLR)
Standard linear regression with probabilistic interpretation.

### Support Vector Regression (SVR)
Non-linear regression using kernel methods (RBF, linear).

### Artificial Neural Network (ANN)
Deep learning model with configurable architecture:
- Default: 3 hidden layers [64, 32, 16]
- Dropout regularization
- Adam optimizer
- Early stopping

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest tests/ -v
```

## Configuration

Key configuration options in `config.yaml`:

```yaml
data:
  input_path: "data/sensor_data.xlsx"
  test_size: 0.2
  random_state: 42

features:
  polynomial_degree: 2
  scaling_method: "standard"
  vif_threshold: 10.0

models:
  ridge:
    enabled: true
    alpha_range: [0.1, 1.0, 10.0, 100.0]
  ann:
    enabled: true
    hidden_layers: [64, 32, 16]
    epochs: 100
```

## Development

### Adding New Models

1. Add model configuration to `config.yaml`
2. Implement model in `src/models.py`:
   - Create pipeline method
   - Add training method
   - Update `train_all_models()`
3. Add tests in `tests/test_models.py`

### Adding New Metrics

1. Implement metric function in `src/metrics.py`
2. Update `evaluate_model()` to include new metric
3. Add tests in `tests/test_metrics.py`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and add tests
4. Run tests (`pytest tests/`)
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to branch (`git push origin feature/new-feature`)
7. Create Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- scikit-learn for ML models and pipelines
- TensorFlow/Keras for deep learning
- pandas and numpy for data manipulation
- matplotlib and seaborn for visualization