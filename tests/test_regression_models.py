import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from models.regression_models import RegressionModels

class TestRegressionModels(unittest.TestCase):
    def setUp(self):
        self.models = RegressionModels()
        # Create simple test data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 3)
        self.y_train = 2 * self.X_train[:, 0] + 3 * self.X_train[:, 1] - self.X_train[:, 2] + np.random.normal(0, 0.1, 100)
        self.X_test = np.random.rand(20, 3)
        self.y_test = 2 * self.X_test[:, 0] + 3 * self.X_test[:, 1] - self.X_test[:, 2] + np.random.normal(0, 0.1, 20)
        
    def test_train_model(self):
        # Test training linear model
        linear_model = self.models.train_model(self.X_train, self.y_train, 'linear')
        self.assertIsNotNone(linear_model)
        
        # Test training ridge model
        ridge_model = self.models.train_model(self.X_train, self.y_train, 'ridge', alpha=0.5)
        self.assertIsNotNone(ridge_model)
        
        # Test training lasso model
        lasso_model = self.models.train_model(self.X_train, self.y_train, 'lasso', alpha=0.1)
        self.assertIsNotNone(lasso_model)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            self.models.train_model(self.X_train, self.y_train, 'invalid_model')
    
    def test_evaluate_model(self):
        # Train a model
        model = self.models.train_model(self.X_train, self.y_train, 'linear')
        
        # Test evaluation
        metrics = self.models.evaluate_model(model, self.X_test, self.y_test)
        
        # Check metrics
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['r2'], 0)  # R² should be positive for this simple data
        
    def test_cross_validate_models(self):
        # Test cross-validation
        cv_results = self.models.cross_validate_models(self.X_train, self.y_train, cv=3)
        
        # Check results structure
        self.assertIn('linear', cv_results)
        self.assertIn('ridge', cv_results)
        self.assertIn('lasso', cv_results)
        
        # Check each model has mean and std scores
        for model_name in ['linear', 'ridge', 'lasso']:
            self.assertIn('mean_score', cv_results[model_name])
            self.assertIn('std_score', cv_results[model_name])

if __name__ == '__main__':
    unittest.main()