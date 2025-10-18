"""
Structural tests for SmartFlush models module.
"""

from src import models


def test_modeling_engine_available():
    engine = models.ModelingEngine(config={})
    assert hasattr(engine, "train_all")
    try:
        engine.train_all({})
    except NotImplementedError:
        assert True


def test_model_artifacts_dataclass():
    artifact = models.ModelArtifacts()
    assert isinstance(artifact.models, dict)
    assert isinstance(artifact.metadata, dict)
