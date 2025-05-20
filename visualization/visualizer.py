import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        self.plt_style = 'seaborn'
        plt.style.use(self.plt_style)

    def plot_correlation_matrix(self, data):
        """Plot correlation matrix of features."""
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        return plt.gcf()

    def plot_feature_importance(self, importance_scores, feature_names):
        """Plot feature importance scores."""
        plt.figure(figsize=(10, 6))
        importance_df = pd.Series(importance_scores, index=feature_names)
        importance_df.sort_values(ascending=True).plot(kind='barh')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        return plt.gcf()

    def plot_predictions_vs_actual(self, y_true, y_pred):
        """Plot predicted vs actual values."""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual House Prices')
        return plt.gcf()

    def plot_residuals(self, y_true, y_pred):
        """Plot residuals distribution."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Count')
        return plt.gcf()

    def plot_model_comparison(self, metrics_dict):
        """Plot comparison of different models' performance."""
        metrics_df = pd.DataFrame(metrics_dict).T
        
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.legend(title='Models')
        plt.xticks(rotation=45)
        return plt.gcf()

    def save_plot(self, figure, filename):
        """Save the plot to a file."""
        figure.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(figure)