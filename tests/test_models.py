"""
Behavioural tests for the SmartFlush modeling engine.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import models


def _build_splits():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(rng.integers(1, 4, size=60), name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1, stratify=y_train
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def test_train_all_returns_predictions():
    config = {
        "grid_search": {"cv_folds": 2, "scoring": "neg_mean_absolute_error", "n_jobs": 1},
        "models": {
            "ridge": {"enabled": True, "grid": {"alpha": [1.0]}},
            "multinomial_logistic": {"enabled": True, "grid": {"classifier__C": [1.0]}},
            "multinomial_logistic_prob": {"enabled": True, "thresholds": [0.5]},
            "svc": {"enabled": False},
            "svc_prob": {"enabled": False},
            "ann": {"enabled": False},
        },
    }

    engine = models.ModelingEngine(config=config)
    data_bundle = {
        "metadata": {"final_feature_names": ["f1", "f2", "f3", "f4"], "removed_vif_features": []},
        "splits": _build_splits(),
    }

    artifacts = engine.train_all(data_bundle)

    assert "ridge" in artifacts.models
    assert "mlr" in artifacts.models
    assert any(name.startswith("mlr_prob") for name in artifacts.models)
    assert artifacts.evaluation_data["predictions"]
    for preds in artifacts.evaluation_data["predictions"].values():
        assert len(preds) == len(artifacts.evaluation_data["y_test"])
