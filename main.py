"""
SmartFlush Predictive Modeling System
Entry point for orchestrating the predictive modeling workflow.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src import data_loading, eda, metrics, models, utils

LOGGER = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="SmartFlush predictive modeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--run-stage",
        type=str,
        default="all",
        choices=["all", "data", "eda", "modeling", "reporting"],
        help="Limit execution to a specific pipeline stage.",
    )
    parser.add_argument(
        "--override-target",
        type=str,
        default=None,
        help="Optional target column override.",
    )
    return parser


def _safe_call(func, *args, **kwargs):
    """Execute a pipeline function while gracefully handling unimplemented sections."""
    try:
        return func(*args, **kwargs)
    except NotImplementedError as exc:
        LOGGER.warning("Skipping %s: %s", func.__name__, exc)
        return None


def _resolve_data_paths(base_dir: Path, file_names: Sequence[str]) -> Sequence[Path]:
    """Resolve data file paths relative to the configured data directory."""
    return [base_dir / Path(name) for name in file_names]


def main(config_path: str = "config.yaml", run_stage: str = "all", override_target: Optional[str] = None) -> None:
    """Run the SmartFlush predictive modeling pipeline."""
    config = utils.load_config(Path(config_path))
    utils.configure_logging(config.get("logging", {}), Path(config.get("paths", {}).get("logs_dir", "logs")))
    LOGGER.info("SmartFlush Predictive Modeling System :: bootstrap")

    paths = utils.ensure_directories(config)

    target_column = override_target or config.get("data", {}).get("target_column")
    data_files = config.get("data", {}).get("files", [])
    data_dir = Path(config.get("paths", {}).get("data_dir", "data"))
    file_paths = _resolve_data_paths(data_dir, data_files)

    raw_dataset = None
    processed_bundle: Optional[Dict[str, Any]] = None

    if run_stage in {"all", "data"}:
        LOGGER.info("Stage: Data loading & preprocessing")
        raw_dataset = _safe_call(data_loading.load_data, file_paths=file_paths, sheet_name=config.get("data", {}).get("sheet_name"))
        if raw_dataset is not None:
            LOGGER.debug("Raw dataset shape: %s", getattr(raw_dataset, "shape", "unknown"))
            processed_bundle = _safe_call(
                data_loading.preprocess_data,
                dataset=raw_dataset,
                target_column=target_column,
                config=config,
            )

    if run_stage in {"all", "eda"} and raw_dataset is not None:
        LOGGER.info("Stage: Exploratory data analysis")
        _safe_call(
            eda.run_eda,
            dataset=raw_dataset,
            target_column=target_column,
            config=config,
            output_directory=paths["figures_dir"],
        )

    modeling_artifacts = None
    if run_stage in {"all", "modeling"} and processed_bundle is not None:
        LOGGER.info("Stage: Modeling & training")
        modeling_engine = models.ModelingEngine(config=config)
        modeling_artifacts = _safe_call(modeling_engine.train_all, processed_bundle)

    if run_stage in {"all", "reporting"} and modeling_artifacts is not None:
        LOGGER.info("Stage: Evaluation & reporting")
        evaluation_results = _safe_call(
            metrics.evaluate_models,
            artifacts=modeling_artifacts,
            config=config,
        )
        if evaluation_results is not None:
            _safe_call(
                metrics.save_reports,
                evaluation_results=evaluation_results,
                output_paths=paths,
            )

    LOGGER.info("SmartFlush pipeline finished")


if __name__ == "__main__":
    cli_args = _build_argument_parser().parse_args()
    main(
        config_path=cli_args.config,
        run_stage=cli_args.run_stage,
        override_target=cli_args.override_target,
    )
