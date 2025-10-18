# Contributing to SmartFlush Predictive Model

Thank you for your interest in contributing to SmartFlush! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/SmartFlush-PredictiveModel.git
cd SmartFlush-PredictiveModel
```

2. **Create a development environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install development dependencies** (optional):
```bash
pip install black flake8 mypy pytest-cov
```

## Code Style Guidelines

### Python Code Style

- Follow **PEP 8** style guide
- Use **type hints** for all function parameters and return values
- Maximum line length: **100 characters**
- Use **docstrings** for all modules, classes, and functions

Example:
```python
def calculate_water_savings(
    actual_levels: np.ndarray,
    predicted_levels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate water savings by comparing actual vs predicted flush levels.
    
    Args:
        actual_levels: Array of actual flush levels
        predicted_levels: Array of predicted flush levels
        
    Returns:
        Dictionary containing savings metrics
        
    Example:
        >>> savings = calculate_water_savings(y_test, predictions)
        >>> print(f"Saved: {savings['savings']}L")
    """
    # Implementation here
    pass
```

### Docstring Format

Use **Google-style docstrings** with:
- Brief one-line summary
- Detailed description (if needed)
- `Args:` section with type information
- `Returns:` section with type information
- `Raises:` section for exceptions
- `Example:` section with usage examples

### Logging

Use the logging module instead of print statements:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting process")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
```

### Error Handling

Always use try/except blocks for operations that might fail:

```python
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test both success and failure cases
- Use pytest markers for slow/integration tests

Example:
```python
import pytest
import numpy as np
from src.utils import calculate_water_usage

class TestWaterCalculations:
    """Tests for water usage calculations."""
    
    def test_calculate_water_usage_valid(self):
        """Test water usage with valid flush levels."""
        flush_levels = np.array([1, 2, 3, 4, 5])
        usage = calculate_water_usage(flush_levels)
        assert usage > 0
        assert isinstance(usage, float)
    
    def test_calculate_water_usage_invalid(self):
        """Test water usage with invalid flush levels."""
        flush_levels = np.array([0, 12, 15])
        with pytest.raises(ValueError):
            calculate_water_usage(flush_levels)
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset."""
        flush_levels = np.random.randint(1, 12, 10000)
        usage = calculate_water_usage(flush_levels)
        assert usage > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::TestWaterCalculations::test_calculate_water_usage_valid

# Run with coverage
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Show print statements
pytest -s
```

### Test Coverage

Aim for **>80% code coverage** for new features. Check coverage with:

```bash
pytest --cov=src --cov-report=term-missing
```

## Adding New Features

### 1. New Model

To add a new machine learning model:

1. Add model implementation to `src/models.py`:
```python
def create_new_model_pipeline(params: List) -> Pipeline:
    """Create pipeline for new model."""
    pass

def train_new_model(X_train, y_train, params) -> Tuple[Any, Dict]:
    """Train new model."""
    pass
```

2. Add tests to `tests/test_models.py`:
```python
class TestNewModel:
    def test_create_new_model_pipeline(self):
        pipeline = create_new_model_pipeline([1, 2, 3])
        assert pipeline is not None
```

3. Update `main.py` to include the new model in the pipeline

4. Add configuration to `config.yaml`:
```yaml
models:
  new_model:
    param1: value1
    param2: value2
```

5. Update documentation in `README.md`

### 2. New Metric

To add a new evaluation metric:

1. Add function to `src/metrics.py`:
```python
def calculate_new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate new metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Metric value
    """
    pass
```

2. Add tests to `tests/test_metrics.py`

3. Update `evaluate_model()` to include the new metric

### 3. New Data Processing Step

To add a new preprocessing step:

1. Add function to `src/data_loading.py`
2. Add tests to `tests/test_data_loading.py`
3. Update `prepare_data()` pipeline if needed
4. Update documentation

## Pull Request Process

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run the test suite**:
```bash
pytest
```

5. **Update documentation** if needed:
   - Update `README.md` for user-facing changes
   - Update docstrings for API changes
   - Add examples if applicable

6. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: brief description"
```

Use clear, descriptive commit messages:
- `feat: Add new water efficiency metric`
- `fix: Correct VIF calculation for edge cases`
- `docs: Update installation instructions`
- `test: Add tests for data loading module`
- `refactor: Simplify model training pipeline`

7. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

8. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Test results
   - Screenshots (if applicable)

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose. The review will check:

- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Breaking changes

## Reporting Issues

When reporting issues, please include:

1. **Description** of the issue
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (OS, Python version, etc.)
6. **Error messages** (if any)

Use the GitHub issue template when available.

## Feature Requests

To request a new feature:

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why it would be valuable
4. Provide examples if possible

## Project Structure

When adding new files, follow this structure:

```
SmartFlush-PredictiveModel/
├── data/              # Data files (not committed)
├── src/               # Source code
│   ├── __init__.py
│   ├── data_loading.py    # Data loading and preprocessing
│   ├── eda.py             # Exploratory data analysis
│   ├── models.py          # ML model implementations
│   ├── metrics.py         # Evaluation metrics
│   └── utils.py           # Utility functions
├── tests/             # Unit tests (mirror src/ structure)
├── notebooks/         # Jupyter notebooks
├── results/           # Output files (not committed)
├── reports/           # Generated reports (not committed)
├── main.py            # Main orchestration script
├── config.yaml        # Configuration
└── README.md          # Documentation
```

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml` and `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Build and publish to PyPI (if applicable)

## Questions?

If you have questions:
- Check existing documentation
- Search closed issues
- Open a new issue with the `question` label
- Reach out to maintainers

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to SmartFlush! 🚀💧
