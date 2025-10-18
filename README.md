# SmartFlush Predictive Modeling System

A commercial-ready Python toolkit for optimizing contactless toilet flush volumes from multi-sensor inputs (urine, paper, feces detectors). The project delivers repeatable data science workflows, transparent governance, and deployment-friendly assets for hospitality, workplace, and smart facility partners.

## Key Capabilities
- Modular data ingestion for multiple Excel sources with preprocessing hooks (VIF screening, residual diagnostics, scaling, polynomial features).
- Experiment-ready EDA utilities covering Chi² associations, Pearson correlation heatmaps, pairplots, and target-focused boxplots.
- Unified modeling interface with scikit-learn pipelines, GridSearchCV, and calibrated prediction adjustments for Ridge, MLR, SVC (linear/poly), and Keras ANN variants.
- Custom evaluation metrics (safe flush accuracy, water efficiency MAE) with benchmark comparisons against a 31% accuracy competitor.
- Scenario analysis for savings in a 100-room hotel (5 flushes/day, 4€/1000L) and reporting pipelines for compliance-ready documentation.

## Project Structure
```
SmartFlush-PredictiveModel-main/
├── data/                # Raw & processed datasets (e.g., Combined_Data.xlsx, mon_fichier.xlsx)
├── notebooks/           # Jupyter notebooks (EDA.ipynb, ANN.ipynb placeholders)
├── reports/             # Generated reports or exported PDFs
├── results/             # Model artifacts, plots, and metric tables
├── src/                 # Reusable source modules
│   ├── __init__.py
│   ├── data_loading.py  # Data ingestion & preprocessing blueprints
│   ├── eda.py           # Exploratory data analysis routines
│   ├── metrics.py       # Custom metrics & benchmark utilities
│   ├── models.py        # Model pipelines & tuning strategies
│   └── utils.py         # Shared helpers (VIF, chi², water savings)
├── tests/               # Pytest-based unit tests
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_eda.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_utils.py
├── config.yaml          # Central configuration (paths, model params, thresholds)
├── main.py              # Orchestrates the end-to-end workflow
├── requirements.txt     # Project dependencies
└── README.md
```

## Getting Started
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Populate data inputs**
   - Drop Excel datasets into `data/` (raw subfolders optional).
   - Update `config.yaml` with filenames, sheet names, and target column metadata.

## Usage
- Run the full pipeline (load ➜ explore ➜ model ➜ evaluate ➜ report):
  ```bash
  python main.py --config config.yaml
  ```
- Generate only EDA assets:
  ```bash
  python main.py --config config.yaml --run-stage eda
  ```
- See `python main.py --help` for staging controls, logging verbosity, and override flags (data paths, target column, threshold tuning).

## Configuration Highlights (`config.yaml`)
- `data`: file list, target column, train/test proportions, random seed, expected sheet names.
- `preprocessing`: VIF threshold, residual feature toggles, polynomial degree, scaler options.
- `models`: parameter grids, threshold search space for probability-based predictions, ANN architecture defaults.
- `evaluation`: competitor baselines, safe flush threshold, hotel savings assumptions.
- `outputs`: directories for plots, tables, serialized pipelines, and PDF summaries.

## Results Tracking
| Model Variant | MAE ↓ | Safe Flush Accuracy ↑ | Water Savings (%) ↑ | Notes |
|---------------|-------|------------------------|---------------------|-------|
| Ridge (+1 adj.) | TBD | TBD | TBD | Deterministic linear baseline |
| MLR (+1 adj.) | TBD | TBD | TBD | Multinomial logistic (class) |
| MLR_2 (prob.) | TBD | TBD | TBD | Tunable probability cutoff |
| SVC (linear/poly) | TBD | TBD | TBD | Avoids RBF for finite feature space |
| SVC_2 (prob.) | TBD | TBD | TBD | Calibrated decision threshold |
| ANN (dense) | TBD | TBD | TBD | ReLU stack + dropout + sigmoid |
| **Competitor** | 0.93 (MAE) | 0.56 | 31% accuracy | Provided industry benchmark |

Populate the table via `results/model_performance.csv` exported from `metrics.save_comparison_table`.

## Testing
```bash
pytest
```
Each test currently validates interface contracts so that implementation can grow iteratively.

## Reporting & Compliance
- `reports/` captures PDF summaries, KPI scorecards, and internal audit artifacts.
- `results/` preserves serialized models, hyperparameter logs, learning curves, and visualization PNGs.
- Logging is centrally configured via `config.yaml` and routed to `logs/` (created on demand).

## Next Steps
- Flesh out the stubbed functions within `src/` to perform the described analytics.
- Extend `tests/` with behavioural checks alongside the structural placeholders shipped here.
- Integrate CI/CD (GitHub Actions, GitLab CI) for linting, testing, and artifact promotion.

## License & Support
- See `LICENSE` for usage terms.
- For commercial engagements or support, open an issue or contact the SmartFlush engineering team.
