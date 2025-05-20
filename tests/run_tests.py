import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_data_processor import TestDataProcessor
from test_regression_models import TestRegressionModels
from test_visualizer import TestVisualizer

def run_all_tests():
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestDataProcessor))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestRegressionModels))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestVisualizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running all tests for House Price Prediction project...\n")
    success = run_all_tests()
    
    if success:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)