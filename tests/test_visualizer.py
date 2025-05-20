import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from visualization.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = Visualizer()
        # Create test data
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'feature3': [2, 3, 4, 5, 6],
            'target': [10, 12, 14, 16, 18]
        })
        
        # Create test metrics
        self.test_metrics = {
            'model1': {'mae': 2.1, 'rmse': 3.2, 'r2': 0.75},
            'model2': {'mae': 1.9, 'rmse': 2.8, 'r2': 0.80},
            'model3': {'mae': 2.3, 'rmse': 3.5, 'r2': 0.72}
        }
        
    def test_plot_correlation_matrix(self):
        # Test correlation matrix plot
        fig = self.visualizer.plot_correlation_matrix(self.test_data)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_plot_feature_importance(self):
        # Create test importance scores
        importance_scores = np.array([0.5, 0.3, 0.2])
        feature_names = ['feature1', 'feature2', 'feature3']
        
        # Test feature importance plot
        fig = self.visualizer.plot_feature_importance(importance_scores, feature_names)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_plot_predictions_vs_actual(self):
        # Create test predictions
        y_true = np.array([10, 12, 14, 16, 18])
        y_pred = np.array([9.5, 12.2, 13.8, 16.5, 17.7])
        
        # Test predictions vs actual plot
        fig = self.visualizer.plot_predictions_vs_actual(y_true, y_pred)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_plot_residuals(self):
        # Create test data for residuals
        y_true = np.array([10, 12, 14, 16, 18])
        y_pred = np.array([9.5, 12.2, 13.8, 16.5, 17.7])
        
        # Test residuals plot
        fig = self.visualizer.plot_residuals(y_true, y_pred)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_plot_model_comparison(self):
        # Test model comparison plot
        fig = self.visualizer.plot_model_comparison(self.test_metrics)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_save_plot(self):
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        
        # Test saving plot
        test_filename = 'test_plot.png'
        self.visualizer.save_plot(fig, test_filename)
        
        # Check if file was created
        self.assertTrue(os.path.exists(test_filename))
        
        # Clean up
        os.remove(test_filename)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()