import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from preprocessing.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'RM': [6.5, 6.4, 7.1],
            'LSTAT': [4.9, 9.1, 4.0],
            'PTRATIO': [15.3, 17.8, 17.8],
            'MEDV': [504000.0, 453600.0, 728700.0]
        })
        
    def test_load_data(self):
        # Create a temporary CSV file
        self.test_data.to_csv('test_housing.csv', index=False)
        
        # Test loading data
        result = self.processor.load_data('test_housing.csv')
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.data)
        self.assertEqual(self.processor.data.shape[0], 3)
        
        # Clean up
        os.remove('test_housing.csv')
        
    def test_preprocess_data(self):
        # Load test data directly
        self.processor.data = self.test_data.copy()
        
        # Test preprocessing
        X, y = self.processor.preprocess_data()
        
        # Check shapes
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(y.shape, (3,))
        
        # Check scaling
        self.assertAlmostEqual(X.mean().mean(), 0, delta=1e-10)
        
    def test_split_data(self):
        # Load and preprocess test data
        self.processor.data = self.test_data.copy()
        self.processor.preprocess_data()
        
        # Test data splitting
        X_train, X_test, y_train, y_test = self.processor.split_data(test_size=0.33)
        
        # Check shapes (with small dataset, might get 2-1 or 3-0 split)
        self.assertEqual(len(X_train) + len(X_test), 3)
        self.assertEqual(len(y_train) + len(y_test), 3)

if __name__ == '__main__':
    unittest.main()