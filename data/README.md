# Data Directory

This directory contains sensor data files for the SmartFlush predictive model.

## Expected Data Format

The predictive model expects Excel files (.xlsx or .xls) with the following structure:

### Required Columns

#### Feature Columns (sensor readings):
- Flow rate measurements
- Pressure measurements  
- Temperature readings
- Duration metrics
- Usage type indicators
- Any other relevant sensor data

#### Target Column:
- `flush_volume` (or custom name): The actual flush volume in liters

### Example Structure

```
| flow_rate | pressure | temperature | duration | usage_type | ... | flush_volume |
|-----------|----------|-------------|----------|------------|-----|--------------|
| 2.5       | 150.0    | 20.5        | 25.0     | 1          | ... | 4.2          |
| 2.8       | 145.0    | 21.0        | 30.0     | 0          | ... | 3.8          |
```

## Creating Sample Data

To generate sample data for testing, run:

```bash
python create_sample_data.py
```

This will create a file `sensor_data.xlsx` with synthetic sensor readings.

## Data Requirements

- Format: Excel (.xlsx or .xls)
- Minimum samples: 100 (recommended: 500+)
- Missing values: Will be handled by data cleaning
- Numeric features: Should be continuous or discrete numeric values
- Target variable: Continuous numeric values (flush volume in liters)

## Notes

- The actual data file (`sensor_data.xlsx`) is excluded from version control via `.gitignore`
- Place your real sensor data in this directory before running the main pipeline
- Ensure feature names are descriptive and match your domain
