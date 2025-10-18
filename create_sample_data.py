"""
Create sample sensor data for testing the pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate synthetic sensor data
data = {
    # Flow rate sensor (L/s)
    'flow_rate': np.random.normal(2.5, 0.5, n_samples),
    
    # Pressure sensor (kPa)
    'pressure': np.random.normal(150, 20, n_samples),
    
    # Temperature sensor (°C)
    'temperature': np.random.normal(20, 3, n_samples),
    
    # Occupancy duration (seconds)
    'duration': np.random.uniform(5, 60, n_samples),
    
    # Usage type (0: small, 1: medium, 2: large)
    'usage_type': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
    
    # Time of day (hour)
    'time_of_day': np.random.randint(0, 24, n_samples),
    
    # Water hardness (ppm)
    'water_hardness': np.random.normal(120, 15, n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target variable (flush_volume) based on features with some noise
# Formula: base volume + factors from sensors
flush_volume = (
    2.0 +  # Base volume
    0.3 * df['flow_rate'] +
    0.01 * df['pressure'] +
    0.05 * df['temperature'] +
    0.02 * df['duration'] +
    0.5 * df['usage_type'] +
    np.random.normal(0, 0.3, n_samples)  # Add noise
)

# Ensure flush volume is within reasonable bounds (2.5 to 6.0 liters)
flush_volume = np.clip(flush_volume, 2.5, 6.0)
df['flush_volume'] = flush_volume

# Add some missing values (5% of data)
mask = np.random.random(n_samples) < 0.05
df.loc[mask, 'water_hardness'] = np.nan

# Create data directory if it doesn't exist
Path('data').mkdir(exist_ok=True)

# Save to Excel
output_file = 'data/sensor_data.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Sample data created successfully!")
print(f"File: {output_file}")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())
