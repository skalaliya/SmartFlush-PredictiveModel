# 🚰 SmartFlush Predictive Modeling System

A vibrant, production-grade ML stack for optimizing contact‑less toilet flush volumes. SmartFlush blends sensor data (urine, paper, feces detectors) with modern data science tooling to deliver confident, water-conscious decisions for hospitality and smart-building partners.

---

## 🌈 Why SmartFlush?

| ⭐ | Capability | What you get |
|----|------------|--------------|
| 💾 | **Modular ingestion** | Combine multi-sheet Excel feeds, normalize headers, and tackle multicollinearity (VIF pruning + residual features) effortlessly. |
| 🔍 | **Rich EDA** | Automated Chi² tests, pearson heatmaps, pairplots, and boxplots saved straight to `results/figures/`. |
| 🤖 | **Model zoo** | Ridge, multinomial logistic (probability-aware variants), SVC (linear/poly), all wrapped in GridSearchCV pipelines with predictable +1 volume offsets. |
| 📊 | **Custom metrics** | Safe flush accuracy, MAE in liters, water/cost savings vs. a 31% competitor baseline, plus confusion matrices & classification reports. |
| 🌍 | **Impact analytics** | Built-in hotel scenario calculators (100 rooms, 5 flushes/day, 4€/1000L) for environmental/economic storytelling. |

---

## 🗂️ Project Map
```
SmartFlush-PredictiveModel-main/
├── data/                # Raw & synthetic Excel datasets
├── notebooks/           # EDA.ipynb, ANN.ipynb (experiments)
├── reports/             # Generated PDFs, briefs
├── results/             # Metrics tables, figures, serialized models
├── src/                 # Core Python modules
│   ├── data_loading.py  # Ingestion + preprocessing pipeline
│   ├── eda.py           # Visualization & statistical diagnostics
│   ├── metrics.py       # Custom metric engines + report writers
│   ├── models.py        # Model orchestration & persistence
│   └── utils.py         # VIF, chi² helpers, impact calculators
├── tests/               # Behavioural pytest suite
├── config.yaml          # Main configuration (use this with real data)
├── config_sample.yaml   # Synthetic sample profile (quick demo)
├── pyproject.toml / uv.lock / .python-version
└── main.py              # CLI orchestration entrypoint
```

---

## 🚀 Quickstart (UV-powered)

```bash
# 1) Create & activate tooling
uv venv --python 3.11

# 2) Install dependencies
uv pip install -r requirements.txt

# 3) Drop Excel files
#    - Put production workbooks into data/
#    - Update config.yaml (files + sheet names + target column)

# 4) Run the full pipeline (load → EDA → model → evaluate → report)
uv run python main.py
```

Need to target a different dataset? Supply a config:

```bash
uv run python main.py --config config_sample.yaml      # synthetic demo
uv run python main.py --config config.yaml --run-stage eda  # EDA only
uv run python main.py --help                           # all runtime flags
```

---

## ⚙️ Configuration Highlights (`config.yaml`)

- **data**: Excel files, sheet selector, target column (`flush_volume_class` in the synthetic set), categorical overrides, split ratios.
- **preprocessing**: Imputation strategy, VIF thresholding, optional residual features, polynomial degree & scaler.
- **models**: Ridge/MLR/SVC grids, probability thresholds, ANN architecture (disabled by default; enable once targets are 0-based).
- **evaluation**: Safe flush threshold, competitor benchmarks, water-cost assumptions, volume mapping.
- **outputs**: Control where figures, tables, and serialized models land (`results/...`).

---

## 📈 Results Dashboard

After each run, check `results/tables/model_summary.csv`. Example from the synthetic dataset:

| Model | Safe Flush ↑ | MAE ↓ (L) | Water Saving ↑ |
|-------|--------------|-----------|----------------|
| `svc_prob_0.55` | 0.87 | 2.30 | 31% |
| `ridge`         | 0.86 | **1.53** | **41%** |
| `mlr`           | 0.79 | 2.27 | 37% |
| `svc`           | 0.68 | 2.48 | 40% |

Confusion matrices & classification reports sit alongside the summary inside `results/tables/`.

---

## ✅ Testing & Quality

```bash
uv run pytest
```

The suite validates:
- Excel ingestion + preprocessing contracts (`tests/test_data_loading.py`)
- EDA artifact generation (`tests/test_eda.py`)
- Custom metric math & reporting (`tests/test_metrics.py`)
- Model orchestration pipelines (`tests/test_models.py`)
- Utility helpers (VIF, chi², impact calculators) (`tests/test_utils.py`)

---

## 📁 Reporting & Compliance Workflow

- `results/figures/`: Heatmaps, pairplots, boxplots, learning curves.
- `results/models/`: Persisted `.joblib` models (if `save_models: true`).
- `reports/`: For publishing PDFs / business briefs.
- Logs routed to `logs/smartflush.log` with formatting driven by `config.yaml`.

---

## 🤝 Contributing & Support

- Pull requests welcome (add tests, keep code documented with docstrings & type hints).
- File issues or reach out for commercial support via GitHub.
- Licensed under the terms in `LICENSE`.

Happy flushing! 💧✨
