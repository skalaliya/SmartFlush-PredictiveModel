# SmartFlush-PredictiveModel

A production-grade predictive model for optimizing water usage in smart flush systems. This project uses machine learning to predict optimal flush levels based on sensor data (photodiodes, waste levels) to minimize water consumption while maintaining hygiene standards.

## Overview

SmartFlush-PredictiveModel implements multiple machine learning approaches to predict flush requirements:
- **Ridge Regression** with polynomial features
- **Multinomial Logistic Regression** with probability threshold optimization
- **Support Vector Classification (SVC)** with linear and polynomial kernels
- **Artificial Neural Networks (ANN)** with dropout regularization

The models analyze sensor inputs to determine the appropriate flush level (1-11), where each level corresponds to a specific water volume, enabling significant water and cost savings.

## Features

- **Multi-model ensemble**: Compare Ridge, LogReg, SVC, and ANN approaches
- **Probability threshold optimization**: Fine-tune classification thresholds for better accuracy
- **VIF-based feature selection**: Handle multicollinearity automatically
- **Comprehensive EDA**: Chi-squared tests, correlation analysis, visualizations
- **Impact calculation**: Estimate water and cost savings for real-world scenarios
- **Production-ready**: Includes logging, error handling, configuration management, and testing

## Project Structure

```
SmartFlush-PredictiveModel/
├── data/                    # Data files (Excel sheets)
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── ANN.ipynb           # ANN model development
├── src/                     # Source code modules
│   ├── data_loading.py     # Data loading and preprocessing
│   ├── eda.py              # Exploratory data analysis
│   ├── models.py           # ML model implementations
│   ├── metrics.py          # Evaluation metrics and benchmarking
│   └── utils.py            # Utility functions
├── reports/                 # Generated reports
├── results/                 # Model outputs and plots
├── tests/                   # Unit tests
├── main.py                  # Main orchestration script
├── config.yaml              # Configuration parameters
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── pytest.ini              # Pytest configuration
└── README.md               # This file
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

### Running the Complete Pipeline

Execute the main script to run the full analysis pipeline:
```bash
python main.py
```

This will:
1. Load and preprocess data from Excel files
2. Perform exploratory data analysis (EDA)
3. Train and evaluate multiple models
4. Optimize probability thresholds
5. Calculate water savings and cost impact
6. Save plots and reports to `results/` and `reports/`

### Running Tests

Execute the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Using Jupyter Notebooks

Launch Jupyter to explore the notebooks:
```bash
jupyter notebook notebooks/
```

- `EDA.ipynb`: Interactive exploratory data analysis
- `ANN.ipynb`: Deep learning model development and experimentation

## Configuration

Edit `config.yaml` to customize:
- Data file paths
- Model hyperparameters
- Threshold optimization ranges
- Hotel scenario parameters (rooms, flushes/day, water costs)
- Logging levels and output paths

## Results

### Model Performance

| Model | Accuracy | Safe Flush Accuracy | MAE | Water Savings |
|-------|----------|---------------------|-----|---------------|
| Competitor Baseline | 0.31 | 0.56 | 0.93 | - |
| Ridge Regression | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |
| SVC (Linear) | TBD | TBD | TBD | TBD |
| SVC (Polynomial) | TBD | TBD | TBD | TBD |
| ANN | TBD | TBD | TBD | TBD |

*Results will be populated after running the pipeline with actual data.*

### Water Savings Impact (100-room hotel scenario)

- **Flushes per day**: 5 per room
- **Annual flushes**: 182,500
- **Water cost**: €4.00 per 1,000L
- **Estimated savings**: TBD L/year, €TBD/year

## Data Requirements

The model expects Excel files with the following features:
- Photodiode sensor readings
- Waste level indicators
- Case of flush (target variable)
- Additional sensor data

Place your data files in the `data/` directory and update paths in `config.yaml`.

## Development

### Adding New Models

1. Implement the model in `src/models.py`
2. Add corresponding tests in `tests/test_models.py`
3. Update `main.py` to include the new model in the pipeline
4. Update configuration in `config.yaml`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sensor data collection and preprocessing
- Water conservation research
- Machine learning optimization techniques