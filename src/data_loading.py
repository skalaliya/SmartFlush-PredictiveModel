"""
Data Loading Module
Handles loading sensor data from Excel files
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess sensor data from Excel files"""
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader
        
        Args:
            file_path: Path to Excel file containing sensor data
        """
        self.file_path = Path(file_path)
        self.data = None
        
    def load_excel(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Args:
            sheet_name: Name of sheet to load (None for first sheet)
            
        Returns:
            DataFrame containing loaded data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be read
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        try:
            logger.info(f"Loading data from {self.file_path}")
            
            # Default to first sheet if sheet_name is None
            if sheet_name is None:
                sheet_name = 0
            
            # Try openpyxl engine first (for .xlsx)
            if self.file_path.suffix == '.xlsx':
                self.data = pd.read_excel(self.file_path, sheet_name=sheet_name, engine='openpyxl')
            # Try xlrd for .xls files
            elif self.file_path.suffix == '.xls':
                self.data = pd.read_excel(self.file_path, sheet_name=sheet_name, engine='xlrd')
            else:
                self.data = pd.read_excel(self.file_path, sheet_name=sheet_name)
                
            logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise ValueError(f"Could not read Excel file: {str(e)}")
    
    def get_data_info(self) -> dict:
        """
        Get information about loaded data
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def clean_data(self, 
                   drop_na: bool = True,
                   drop_duplicates: bool = True,
                   fill_method: Optional[str] = None) -> pd.DataFrame:
        """
        Clean loaded data
        
        Args:
            drop_na: Whether to drop rows with missing values
            drop_duplicates: Whether to drop duplicate rows
            fill_method: Method to fill missing values ('mean', 'median', 'mode', None)
            
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_excel() first.")
            
        cleaned_data = self.data.copy()
        initial_rows = len(cleaned_data)
        
        # Handle missing values
        if fill_method:
            if fill_method == 'mean':
                cleaned_data = cleaned_data.fillna(cleaned_data.mean(numeric_only=True))
            elif fill_method == 'median':
                cleaned_data = cleaned_data.fillna(cleaned_data.median(numeric_only=True))
            elif fill_method == 'mode':
                cleaned_data = cleaned_data.fillna(cleaned_data.mode().iloc[0])
        elif drop_na:
            cleaned_data = cleaned_data.dropna()
            
        # Drop duplicates
        if drop_duplicates:
            cleaned_data = cleaned_data.drop_duplicates()
            
        rows_removed = initial_rows - len(cleaned_data)
        logger.info(f"Cleaned data: removed {rows_removed} rows")
        
        self.data = cleaned_data
        return cleaned_data
    
    def split_features_target(self, 
                             target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Args:
            target_column: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_excel() first.")
            
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        logger.info(f"Split data: {X.shape[1]} features, {len(y)} samples")
        
        return X, y


def load_data_from_excel(file_path: str, 
                         target_column: str,
                         clean: bool = True,
                         sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and prepare data from Excel
    
    Args:
        file_path: Path to Excel file
        target_column: Name of target column
        clean: Whether to clean data
        sheet_name: Name of sheet to load
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    loader = DataLoader(file_path)
    loader.load_excel(sheet_name=sheet_name)
    
    if clean:
        loader.clean_data()
        
    return loader.split_features_target(target_column)
