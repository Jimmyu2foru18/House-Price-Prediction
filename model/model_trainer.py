import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42)
        }
        self.best_model = None

    def train(self, model_name, X_train, y_train):
        """Train specified model on training data"""
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        """Generate predictions using trained model"""
        return model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'r2': r2_score(y_true, y_pred)
        }

    def tune_hyperparameters(self, model_name, X_train, y_train, param_grid):
        """Perform hyperparameter tuning using GridSearchCV"""
        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_

    def save_model(self, model, file_path):
        """Save trained model to disk"""
        joblib.dump(model, file_path)

    def load_model(self, file_path):
        """Load pretrained model from disk"""
        return joblib.load(file_path)