"""
Exploratory Data Analysis (EDA) module for SmartFlush Predictive Model.

This module provides functions for:
- Chi-squared tests for categorical features
- Pearson correlation analysis
- Correlation heatmaps
- Pairplot visualizations
- Boxplots for features vs targets (photodiodes, case of flush, waste levels)
- Distribution analysis
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from src.utils import chi_squared_test

# Configure logging
logger = logging.getLogger(__name__)


def set_plot_style(style: str = 'seaborn-v0_8-darkgrid', context: str = 'notebook') -> None:
    """
    Set the plotting style for consistency across visualizations.
    
    Args:
        style: Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        context: Seaborn context (default: 'notebook')
    """
    try:
        plt.style.use(style)
        sns.set_context(context)
        logger.info(f"Plot style set to: {style}, context: {context}")
    except Exception as e:
        logger.warning(f"Could not set plot style {style}: {e}")


def calculate_pearson_correlation(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate Pearson correlation coefficients between features.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns (if None, uses all numeric columns)
        
    Returns:
        Correlation matrix as DataFrame
        
    Example:
        >>> corr_matrix = calculate_pearson_correlation(df)
    """
    try:
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Calculating Pearson correlation for {len(feature_cols)} features")
        
        corr_matrix = df[feature_cols].corr(method='pearson')
        
        logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
        return corr_matrix
        
    except Exception as e:
        logger.error(f"Error calculating Pearson correlation: {e}")
        raise


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    cmap: str = 'coolwarm',
    title: str = 'Feature Correlation Heatmap'
) -> None:
    """
    Create and save a correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        save_path: Path to save the figure (if None, displays only)
        figsize: Figure size (width, height)
        annot: Whether to annotate cells with correlation values
        cmap: Colormap name
        title: Plot title
        
    Example:
        >>> corr = calculate_pearson_correlation(df)
        >>> plot_correlation_heatmap(corr, save_path='results/correlation_heatmap.png')
    """
    try:
        logger.info(f"Creating correlation heatmap")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        raise


def plot_pairplot(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    save_path: Optional[Path] = None,
    sample_size: Optional[int] = 1000
) -> None:
    """
    Create pairplot to visualize relationships between features.
    
    Args:
        df: Input DataFrame
        feature_cols: List of features to include in pairplot
        target_col: Optional target column for color coding
        save_path: Path to save the figure
        sample_size: If specified, sample this many rows for faster plotting
        
    Example:
        >>> plot_pairplot(df, ['feature1', 'feature2', 'feature3'], target_col='flush_level')
    """
    try:
        logger.info(f"Creating pairplot for {len(feature_cols)} features")
        
        # Sample data if needed for performance
        plot_df = df.copy()
        if sample_size and len(plot_df) > sample_size:
            plot_df = plot_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows for pairplot")
        
        # Select columns for plotting
        cols_to_plot = feature_cols.copy()
        if target_col and target_col not in cols_to_plot:
            cols_to_plot.append(target_col)
        
        # Create pairplot
        if target_col:
            pairplot = sns.pairplot(
                plot_df[cols_to_plot],
                hue=target_col,
                diag_kind='kde',
                plot_kws={'alpha': 0.6},
                diag_kws={'alpha': 0.7}
            )
        else:
            pairplot = sns.pairplot(
                plot_df[cols_to_plot],
                diag_kind='kde',
                plot_kws={'alpha': 0.6},
                diag_kws={'alpha': 0.7}
            )
        
        pairplot.fig.suptitle('Feature Pairplot', y=1.01, fontsize=16, fontweight='bold')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pairplot.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pairplot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating pairplot: {e}")
        raise


def plot_boxplots(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create boxplots showing feature distributions across target categories.
    
    Args:
        df: Input DataFrame
        features: List of feature columns to plot
        target_col: Target column for grouping
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        
    Example:
        >>> plot_boxplots(df, ['photodiode1', 'photodiode2'], target_col='flush_level')
    """
    try:
        logger.info(f"Creating boxplots for {len(features)} features vs {target_col}")
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is 2D array
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, feature in enumerate(features):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Create boxplot
            df.boxplot(column=feature, by=target_col, ax=ax)
            ax.set_title(f'{feature} by {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel(feature)
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # Hide unused subplots
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Feature Distributions by {target_col}', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplots saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating boxplots: {e}")
        raise


def perform_chi_squared_analysis(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Perform Chi-squared tests for categorical features vs target.
    
    Args:
        df: Input DataFrame
        categorical_features: List of categorical feature columns
        target_col: Target column name
        save_path: Optional path to save results as CSV
        
    Returns:
        DataFrame with Chi-squared test results
        
    Example:
        >>> chi2_results = perform_chi_squared_analysis(
        ...     df, ['category1', 'category2'], 'flush_level'
        ... )
    """
    try:
        logger.info(f"Performing Chi-squared analysis for {len(categorical_features)} features")
        
        results = []
        
        for feature in categorical_features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in DataFrame, skipping")
                continue
            
            try:
                chi2_stat, p_value, contingency_table = chi_squared_test(df, feature, target_col)
                
                results.append({
                    'Feature': feature,
                    'Chi2_Statistic': chi2_stat,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
                
            except Exception as e:
                logger.warning(f"Could not perform Chi-squared test for {feature}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Chi2_Statistic', ascending=False)
        
        logger.info(f"Chi-squared analysis complete. "
                   f"{results_df['Significant'].sum()} significant features (p < 0.05)")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)
            logger.info(f"Chi-squared results saved to {save_path}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error performing Chi-squared analysis: {e}")
        raise


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        
    Example:
        >>> plot_target_distribution(df, 'flush_level', 'results/target_dist.png')
    """
    try:
        logger.info(f"Plotting distribution of target: {target_col}")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        target_counts = df[target_col].value_counts().sort_index()
        axes[0].bar(target_counts.index, target_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Target Distribution (Counts)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Percentage plot
        target_pct = (target_counts / len(df) * 100)
        axes[1].bar(target_pct.index, target_pct.values, color='coral', alpha=0.7)
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Target Distribution (Percentage)')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{target_col} Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Target distribution plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting target distribution: {e}")
        raise


def generate_summary_statistics(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save statistics as CSV
        
    Returns:
        DataFrame with summary statistics
        
    Example:
        >>> stats = generate_summary_statistics(df, 'reports/summary_stats.csv')
    """
    try:
        logger.info("Generating summary statistics")
        
        # Basic statistics
        summary = df.describe(include='all').T
        
        # Add additional statistics
        summary['missing_count'] = df.isnull().sum()
        summary['missing_percentage'] = (df.isnull().sum() / len(df) * 100).round(2)
        summary['dtype'] = df.dtypes
        
        logger.info(f"Summary statistics generated for {len(summary)} columns")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(save_path)
            logger.info(f"Summary statistics saved to {save_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")
        raise


def perform_eda(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of numeric feature columns (if None, auto-detect)
        categorical_features: List of categorical features (if None, skip Chi-squared)
        output_dir: Directory to save all plots and results
        
    Returns:
        Dictionary containing all EDA results
        
    Example:
        >>> eda_results = perform_eda(
        ...     df, 
        ...     target_col='flush_level',
        ...     output_dir=Path('results/eda')
        ... )
    """
    try:
        logger.info("Starting comprehensive EDA")
        
        results = {}
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # Set output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary statistics
        logger.info("Generating summary statistics")
        summary_path = output_dir / 'summary_statistics.csv' if output_dir else None
        results['summary_stats'] = generate_summary_statistics(df, summary_path)
        
        # 2. Target distribution
        logger.info("Plotting target distribution")
        target_path = output_dir / 'target_distribution.png' if output_dir else None
        plot_target_distribution(df, target_col, target_path)
        
        # 3. Correlation analysis
        logger.info("Calculating correlations")
        results['correlation_matrix'] = calculate_pearson_correlation(df, feature_cols)
        
        corr_path = output_dir / 'correlation_heatmap.png' if output_dir else None
        plot_correlation_heatmap(results['correlation_matrix'], corr_path)
        
        # 4. Chi-squared analysis for categorical features
        if categorical_features:
            logger.info("Performing Chi-squared analysis")
            chi2_path = output_dir / 'chi_squared_results.csv' if output_dir else None
            results['chi_squared'] = perform_chi_squared_analysis(
                df, categorical_features, target_col, chi2_path
            )
        
        # 5. Boxplots (use subset of features to avoid overcrowding)
        if len(feature_cols) > 0:
            logger.info("Creating boxplots")
            boxplot_features = feature_cols[:6]  # Limit to first 6 for clarity
            boxplot_path = output_dir / 'feature_boxplots.png' if output_dir else None
            plot_boxplots(df, boxplot_features, target_col, boxplot_path)
        
        # 6. Pairplot (use subset of features for performance)
        if len(feature_cols) >= 2:
            logger.info("Creating pairplot")
            pairplot_features = feature_cols[:5]  # Limit to first 5 for performance
            pairplot_path = output_dir / 'feature_pairplot.png' if output_dir else None
            plot_pairplot(df, pairplot_features, target_col, pairplot_path, sample_size=500)
        
        logger.info("EDA complete")
        return results
        
    except Exception as e:
        logger.error(f"Error performing EDA: {e}")
        raise
