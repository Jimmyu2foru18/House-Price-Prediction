# House Price Prediction Project

## Overview
An advanced machine learning project that predicts Boston housing prices using regression models. 
The system analyzes various features including room count, neighborhood demographics, 
and school quality to provide accurate price estimations.

## Project Structure
```
├── data_preprocessing/   # Additional preprocessing utilities
├── model/                # Model training utilities
├── models/               # ML model implementations
├── preprocessing/        # Data preprocessing modules
├── src/                  # Main application code
├── tests/                # Unit tests
├── visualization/        # Plotting and visualization code
├── housing.csv           # Boston housing dataset
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Features
- Data preprocessing pipeline
- Regression models (Linear/Ridge/Lasso)
- Interactive dashboard
- Model evaluation metrics
- Cross-validation

## Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment

## Installation
1. Clone the repository
```bash
git clone https://github.com/jimmyu2foru18/house-price-prediction.git
cd house-price-prediction
```

2. Create and activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset Setup
1. Download the Boston Housing Dataset from [Kaggle](https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd)
2. Place the downloaded CSV file in the `root/` directory
3. Rename the file to `housing.csv` if necessary

## Usage

### Running the Complete Pipeline
```bash
# Run the complete pipeline (data preprocessing, model training, evaluation, and visualization)
python run_pipeline.py
```
This will process the data, train all models, evaluate their performance, and generate visualizations in the `output` directory.

### Data Preprocessing
```python
from preprocessing.data_processor import DataProcessor

# Initialize preprocessor
processor = DataProcessor()

# Load and preprocess data
processor.load_data('housing.csv')
X, y = processor.preprocess_data()
X_train, X_test, y_train, y_test = processor.split_data()
```

### Model Training
```python
from models.regression_models import RegressionModels

# Initialize models
models = RegressionModels()

# Train a specific model
linear_model = models.train_model(X_train, y_train, 'linear')
ridge_model = models.train_model(X_train, y_train, 'ridge', alpha=0.5)
lasso_model = models.train_model(X_train, y_train, 'lasso', alpha=0.1)

# Evaluate model
metrics = models.evaluate_model(model, X_test, y_test)
print(f"R² Score: {metrics['r2']:.4f}")
```

### Visualization
```python
from visualization.visualizer import Visualizer

# Initialize visualizer
visualizer = Visualizer()

# Create visualizations
correlation_plot = visualizer.plot_correlation_matrix(data)
visualizer.save_plot(correlation_plot, 'correlation_matrix.png')

prediction_plot = visualizer.plot_predictions_vs_actual(y_test, y_pred)
visualizer.save_plot(prediction_plot, 'predictions_vs_actual.png')
```

## Model Performance
### Evaluation Metrics
Based on the latest run of the pipeline, here are the actual model performance metrics:

| Metric | Linear | Ridge | Lasso |
|--------|--------|-------|-------|
| MAE    | ~64277 | ~64277| ~64277|
| RMSE   | ~82396 | ~82396| ~82396|
| R²     | 0.69   | 0.69  | 0.69  |

### Visualizations
The pipeline generates the following visualizations in the `output` directory:

- `correlation_matrix.png` - Correlation between different features
- `feature_importance.png` - Importance of each feature for prediction
- `model_comparison.png` - Comparison of different models' performance
- `linear_predictions.png` - Predicted vs actual values for the best model
- `linear_residuals.png` - Distribution of prediction errors

## Development
### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Run individual test modules
python tests/test_data_processor.py
python tests/test_regression_models.py
python tests/test_visualizer.py
```

### Code Style
We follow PEP 8 guidelines. Run linter:
```bash
flake8 ../
```

## Acknowledgments
- Boston Housing Dataset from Kaggle
- scikit-learn documentation and community