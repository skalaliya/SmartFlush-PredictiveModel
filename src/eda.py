"""
Exploratory Data Analysis Module
Includes Chi-square tests, Pearson correlation, and visualization
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
from scipy.stats import chi2_contingency, pearsonr
from pathlib import Path

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Perform exploratory data analysis on sensor data"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, config: dict = None):
        """
        Initialize EDA Analyzer
        
        Args:
            X: Feature DataFrame
            y: Target Series
            config: Configuration dictionary
        """
        self.X = X
        self.y = y
        self.config = config or {}
        self.eda_config = self.config.get('eda', {})
        self.figsize = tuple(self.eda_config.get('plot_figsize', [12, 8]))
        
    def chi_square_test(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform Chi-square test for categorical features
        
        Args:
            alpha: Significance level
            
        Returns:
            DataFrame with chi-square test results
        """
        logger.info("Performing Chi-square tests")
        
        results = []
        
        for column in self.X.columns:
            try:
                # Create contingency table
                contingency_table = pd.crosstab(self.X[column], self.y)
                
                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                results.append({
                    'feature': column,
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < alpha
                })
            except Exception as e:
                logger.warning(f"Could not perform chi-square test for {column}: {str(e)}")
                
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            logger.info(f"Chi-square tests complete: {results_df['significant'].sum()} significant features")
        
        return results_df
    
    def pearson_correlation(self, threshold: float = 0.7) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Calculate Pearson correlation coefficients
        
        Args:
            threshold: Correlation threshold for high correlation
            
        Returns:
            Tuple of (correlation matrix, list of highly correlated pairs)
        """
        logger.info("Calculating Pearson correlations")
        
        # Correlation matrix
        corr_matrix = self.X.corr(method='pearson')
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        logger.info(f"Found {len(high_corr_pairs)} pairs with |correlation| >= {threshold}")
        
        return corr_matrix, high_corr_pairs
    
    def target_correlation(self) -> pd.Series:
        """
        Calculate correlation between features and target
        
        Returns:
            Series with correlations sorted by absolute value
        """
        logger.info("Calculating target correlations")
        
        correlations = {}
        
        for column in self.X.columns:
            try:
                # Calculate Pearson correlation
                corr, p_value = pearsonr(self.X[column], self.y)
                correlations[column] = corr
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {column}: {str(e)}")
                correlations[column] = np.nan
        
        corr_series = pd.Series(correlations)
        corr_series = corr_series.sort_values(ascending=False, key=abs)
        
        return corr_series
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Plot correlation heatmap
        
        Args:
            save_path: Path to save plot (optional)
        """
        logger.info("Generating correlation heatmap")
        
        plt.figure(figsize=self.figsize)
        
        corr_matrix = self.X.corr()
        
        sns.heatmap(corr_matrix, 
                   annot=True if len(self.X.columns) <= 10 else False,
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        plt.close()
    
    def plot_target_correlation(self, save_path: Optional[str] = None):
        """
        Plot correlation with target variable
        
        Args:
            save_path: Path to save plot (optional)
        """
        logger.info("Generating target correlation plot")
        
        correlations = self.target_correlation()
        
        plt.figure(figsize=self.figsize)
        
        colors = ['green' if x > 0 else 'red' for x in correlations.values]
        plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('Correlation with Target', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature-Target Correlations', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Target correlation plot saved to {save_path}")
        
        plt.close()
    
    def plot_distributions(self, save_path: Optional[str] = None):
        """
        Plot feature distributions
        
        Args:
            save_path: Path to save plot (optional)
        """
        logger.info("Generating distribution plots")
        
        n_features = len(self.X.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, column in enumerate(self.X.columns):
            if idx < len(axes):
                axes[idx].hist(self.X[column], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                axes[idx].set_title(column, fontsize=10)
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {save_path}")
        
        plt.close()
    
    def plot_pairplot(self, n_features: int = 5, save_path: Optional[str] = None):
        """
        Plot pairplot for top correlated features
        
        Args:
            n_features: Number of top features to include
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating pairplot for top {n_features} features")
        
        # Get top correlated features with target
        top_features = self.target_correlation().head(n_features).index.tolist()
        
        # Create combined dataframe
        plot_data = self.X[top_features].copy()
        plot_data['target'] = self.y
        
        # Create pairplot
        pairplot = sns.pairplot(plot_data, hue='target', diag_kind='kde', 
                                plot_kws={'alpha': 0.6}, height=2.5)
        pairplot.fig.suptitle(f'Pairplot of Top {n_features} Features', 
                             fontsize=14, fontweight='bold', y=1.01)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pairplot saved to {save_path}")
        
        plt.close()
    
    def generate_summary_report(self) -> dict:
        """
        Generate summary statistics report
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary report")
        
        report = {
            'n_samples': len(self.X),
            'n_features': len(self.X.columns),
            'feature_names': list(self.X.columns),
            'target_name': self.y.name if hasattr(self.y, 'name') else 'target',
            'missing_values': self.X.isnull().sum().to_dict(),
            'feature_stats': self.X.describe().to_dict(),
            'target_stats': self.y.describe().to_dict()
        }
        
        return report
    
    def run_full_eda(self, output_dir: str = "reports/figures/"):
        """
        Run complete EDA pipeline with all plots
        
        Args:
            output_dir: Directory to save plots
        """
        logger.info("Running full EDA pipeline")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_correlation_heatmap(save_path=str(output_path / "correlation_heatmap.png"))
        self.plot_target_correlation(save_path=str(output_path / "target_correlation.png"))
        self.plot_distributions(save_path=str(output_path / "distributions.png"))
        
        # Chi-square and Pearson tests
        chi2_results = self.chi_square_test()
        corr_matrix, high_corr = self.pearson_correlation()
        
        # Save results
        if len(chi2_results) > 0:
            chi2_results.to_csv(output_path / "chi_square_results.csv", index=False)
        corr_matrix.to_csv(output_path / "correlation_matrix.csv")
        
        logger.info("Full EDA complete")


def perform_eda(X: pd.DataFrame, 
                y: pd.Series,
                config: dict = None,
                output_dir: str = "reports/figures/") -> dict:
    """
    Convenience function to perform complete EDA
    
    Args:
        X: Feature DataFrame
        y: Target Series
        config: Configuration dictionary
        output_dir: Output directory for plots
        
    Returns:
        Summary report dictionary
    """
    analyzer = EDAAnalyzer(X, y, config)
    analyzer.run_full_eda(output_dir)
    return analyzer.generate_summary_report()
