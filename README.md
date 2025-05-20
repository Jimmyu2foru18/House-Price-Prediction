# House Price Prediction Project

## Overview
An advanced machine learning project that predicts Boston housing prices using regression models. 
The system analyzes various features including room count, neighborhood demographics, 
and school quality to provide accurate price estimations.

## Project Structure
```
├── data/                 # Dataset storage
├── preprocessing/        # Data preprocessing modules
├── models/               # ML model implementations
└── visualization/        # Plotting and dashboard code
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for analysis
└── docs/                 # Documentation
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
### Data Preprocessing
```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor('data/housing.csv')

# Clean and prepare data
df = preprocessor.clean_data()
X_train, X_test, y_train, y_test = preprocessor.split_data()
```

### Model Training
```bash
# Train a specific model with hyperparameters
python train_model.py --model ridge --alpha 0.5

# Train all models with default parameters
python train_model.py --all
```

### Interactive Dashboard
```bash
python run_dashboard.py
```
Then open `http://localhost:8000` in your browser

## Model Performance
### Evaluation Metrics
| Metric | Baseline | Ridge | Lasso |
|--------|----------|-------|-------|
| MAE    | 3.2      | 2.9   | 3.1   |
| RMSE   | 5.1      | 4.7   | 4.9   |
| R²     | 0.74     | 0.78  | 0.76  |

## Development
### Running Tests
```bash
python -m pytest tests/
```

### Code Style
We follow PEP 8 guidelines. Run linter:
```bash
flake8 ../
```

## Acknowledgments
- Boston Housing Dataset from Kaggle
- scikit-learn documentation and community