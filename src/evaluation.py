"""
Evaluation Module for Agricultural Price Prediction Models

This module implements comprehensive evaluation utilities including:
- Multiple regression metrics (RMSE, MAE, MAPE, R²)
- Prediction visualization
- Model comparison analysis
- Residual diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List, Tuple, Optional, Any
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation for time series forecasting.
    
    Metrics:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² Score (Coefficient of Determination)
    - Directional Accuracy
    """
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'lstm': '#1f77b4',
            'gru': '#2ca02c',
            'transformer': '#d62728',
            'ensemble': '#9467bd',
            'actual': '#333333'
        }
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metric names to values
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle zeros)
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy (trend prediction)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_acc = np.mean(true_direction == pred_direction) * 100
        else:
            directional_acc = 0.0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_acc
        }
        
        return metrics
    
    def evaluate_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and create comparison.
        
        Args:
            y_true: Actual values
            predictions: Dictionary of model name to predictions
            
        Returns:
            DataFrame with metrics for each model
        """
        results = []
        
        for model_name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            metrics['Model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        
        return df
    
    def print_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model'
    ):
        """Print formatted metrics for a model."""
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        print(f"\n{'='*50}")
        print(f" {model_name} Evaluation Metrics")
        print(f"{'='*50}")
        print(f"  RMSE:                {metrics['RMSE']:.4f}")
        print(f"  MAE:                 {metrics['MAE']:.4f}")
        print(f"  MAPE:                {metrics['MAPE']:.2f}%")
        print(f"  R² Score:            {metrics['R2']:.4f}")
        print(f"  Directional Acc:     {metrics['Directional_Accuracy']:.2f}%")
        print(f"{'='*50}\n")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model',
        dates: np.ndarray = None,
        title: str = None,
        save: bool = True,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            dates: Date labels for x-axis
            title: Plot title
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(y_true))
        
        ax.plot(x, y_true, label='Actual', color=self.colors['actual'],
                linewidth=2, alpha=0.8)
        
        color = self.colors.get(model_name.lower(), '#1f77b4')
        ax.plot(x, y_pred, label=f'{model_name} Predicted', color=color,
                linewidth=2, linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Price (INR per Quintal)', fontsize=12)
        ax.set_title(title or f'{model_name}: Actual vs Predicted Prices', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name.lower()}_predictions.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig
    
    def plot_multi_model_predictions(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = 'Model Comparison: Actual vs Predicted',
        save: bool = True,
        figsize: Tuple[int, int] = (16, 8)
    ) -> plt.Figure:
        """
        Plot predictions from multiple models.
        
        Args:
            y_true: Actual values
            predictions: Dictionary of model name to predictions
            title: Plot title
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(y_true))
        
        # Plot actual
        ax.plot(x, y_true, label='Actual', color=self.colors['actual'],
                linewidth=2.5, alpha=0.9)
        
        # Plot each model's predictions
        for model_name, y_pred in predictions.items():
            color = self.colors.get(model_name.lower(), None)
            ax.plot(x, y_pred, label=f'{model_name}', color=color,
                    linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Price (INR per Quintal)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'multi_model_predictions.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        save: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Create bar charts comparing model metrics.
        
        Args:
            metrics_df: DataFrame with metrics for each model
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
        titles = ['RMSE (Lower is better)', 'MAE (Lower is better)',
                  'MAPE % (Lower is better)', 'R² Score (Higher is better)',
                  'Directional Accuracy % (Higher is better)']
        
        colors = [self.colors.get(model.lower(), '#1f77b4') 
                  for model in metrics_df.index]
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            values = metrics_df[metric].values
            models = metrics_df.index.tolist()
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        axes[-1].axis('off')
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model',
        save: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Create residual diagnostic plots.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals, color='steelblue', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time', fontsize=11)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Residual')
        
        # 2. Residual distribution
        axes[0, 1].hist(residuals, bins=30, color='steelblue', alpha=0.7,
                        edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Residual Distribution', fontsize=11)
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Predicted vs Residual
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='steelblue', s=20)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Predicted vs Residuals', fontsize=11)
        axes[1, 0].set_xlabel('Predicted Value')
        axes[1, 0].set_ylabel('Residual')
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=11)
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=14, 
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name.lower()}_residuals.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig
    
    def plot_training_history(
        self,
        histories: Dict[str, Any],
        metric: str = 'loss',
        save: bool = True,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot training and validation curves for multiple models.
        
        Args:
            histories: Dictionary of model name to training history
            metric: Metric to plot ('loss', 'mae', 'mse')
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Training curves
        ax = axes[0]
        for model_name, history in histories.items():
            color = self.colors.get(model_name.lower(), None)
            if hasattr(history, 'history'):
                hist = history.history
            else:
                hist = history
            
            epochs = range(1, len(hist[metric]) + 1)
            ax.plot(epochs, hist[metric], label=model_name, color=color, linewidth=2)
        
        ax.set_title(f'Training {metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation curves
        ax = axes[1]
        for model_name, history in histories.items():
            color = self.colors.get(model_name.lower(), None)
            if hasattr(history, 'history'):
                hist = history.history
            else:
                hist = history
            
            val_metric = f'val_{metric}'
            if val_metric in hist:
                epochs = range(1, len(hist[val_metric]) + 1)
                ax.plot(epochs, hist[val_metric], label=model_name, color=color, linewidth=2)
        
        ax.set_title(f'Validation {metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'training_history_{metric}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig
    
    def create_summary_report(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        histories: Dict[str, Any] = None
    ) -> str:
        """
        Create a comprehensive text summary report.
        
        Args:
            y_true: Actual values
            predictions: Dictionary of model name to predictions
            histories: Dictionary of model name to training history
            
        Returns:
            Summary report string
        """
        metrics_df = self.evaluate_models(y_true, predictions)
        
        report = []
        report.append("="*60)
        report.append(" AGRICULTURAL COMMODITY PRICE PREDICTION - MODEL EVALUATION")
        report.append("="*60)
        report.append("")
        
        # Best model per metric
        report.append("BEST MODEL BY METRIC:")
        report.append("-"*40)
        
        best_rmse = metrics_df['RMSE'].idxmin()
        best_mae = metrics_df['MAE'].idxmin()
        best_mape = metrics_df['MAPE'].idxmin()
        best_r2 = metrics_df['R2'].idxmax()
        best_dir = metrics_df['Directional_Accuracy'].idxmax()
        
        report.append(f"  RMSE:               {best_rmse} ({metrics_df.loc[best_rmse, 'RMSE']:.4f})")
        report.append(f"  MAE:                {best_mae} ({metrics_df.loc[best_mae, 'MAE']:.4f})")
        report.append(f"  MAPE:               {best_mape} ({metrics_df.loc[best_mape, 'MAPE']:.2f}%)")
        report.append(f"  R² Score:           {best_r2} ({metrics_df.loc[best_r2, 'R2']:.4f})")
        report.append(f"  Directional Acc:    {best_dir} ({metrics_df.loc[best_dir, 'Directional_Accuracy']:.2f}%)")
        report.append("")
        
        # Detailed metrics table
        report.append("DETAILED METRICS BY MODEL:")
        report.append("-"*40)
        report.append(metrics_df.to_string())
        report.append("")
        
        # Overall recommendation
        report.append("RECOMMENDATION:")
        report.append("-"*40)
        
        # Simple scoring: count wins
        wins = {}
        for model in metrics_df.index:
            wins[model] = 0
            if model == best_rmse: wins[model] += 1
            if model == best_mae: wins[model] += 1
            if model == best_mape: wins[model] += 1
            if model == best_r2: wins[model] += 1
            if model == best_dir: wins[model] += 1
        
        best_overall = max(wins, key=wins.get)
        report.append(f"  Best Overall Model: {best_overall} (won {wins[best_overall]}/5 metrics)")
        report.append("")
        report.append("="*60)
        
        summary = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, '../reports/evaluation_report.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(summary)
        print(f"✓ Saved report: {report_path}")
        
        return summary
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        save: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance scores
            top_n: Number of top features to show
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, importances[indices], align='center', 
                color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {filepath}")
        
        return fig


def evaluate_all_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    histories: Dict[str, Any] = None,
    output_dir: str = 'outputs/figures'
) -> Tuple[pd.DataFrame, str]:
    """
    Comprehensive evaluation of all models.
    
    Args:
        y_true: Actual values
        predictions: Dictionary of model name to predictions
        histories: Dictionary of model name to training history
        output_dir: Directory for saving outputs
        
    Returns:
        Tuple of (metrics DataFrame, summary report string)
    """
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Print metrics for each model
    for model_name, y_pred in predictions.items():
        evaluator.print_metrics(y_true, y_pred, model_name)
    
    # Create comparison DataFrame
    metrics_df = evaluator.evaluate_models(y_true, predictions)
    print("\nModel Comparison Summary:")
    print(metrics_df.to_string())
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Individual predictions
    for model_name, y_pred in predictions.items():
        evaluator.plot_predictions(y_true, y_pred, model_name)
        evaluator.plot_residuals(y_true, y_pred, model_name)
    
    # Multi-model comparison
    evaluator.plot_multi_model_predictions(y_true, predictions)
    evaluator.plot_metrics_comparison(metrics_df)
    
    # Training history
    if histories:
        evaluator.plot_training_history(histories, metric='loss')
        evaluator.plot_training_history(histories, metric='mae')
    
    # Summary report
    summary = evaluator.create_summary_report(y_true, predictions, histories)
    print(summary)
    
    return metrics_df, summary
