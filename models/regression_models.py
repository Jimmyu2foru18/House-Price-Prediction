from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class RegressionModels:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.ridge_model = Ridge()
        self.lasso_model = Lasso()
        self.models = {
            'linear': self.linear_model,
            'ridge': self.ridge_model,
            'lasso': self.lasso_model
        }

    def train_model(self, X_train, y_train, model_type='linear', **kwargs):
        """Train the specified regression model."""
        if model_type not in self.models:
            raise ValueError(f"Invalid model type. Choose from {list(self.models.keys())}")

        # Update model parameters if provided
        if kwargs and model_type in ['ridge', 'lasso']:
            self.models[model_type].set_params(**kwargs)

        # Train the model
        model = self.models[model_type]
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model using multiple metrics."""
        y_pred = model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        return metrics

    def get_feature_coefficients(self, model, feature_names):
        """Get the coefficients/importance of each feature."""
        if not hasattr(model, 'coef_'):
            raise ValueError("Model does not have feature coefficients.")

        coefficients = pd.Series(model.coef_, index=feature_names)
        return coefficients.sort_values(ascending=False)

    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation for all models."""
        from sklearn.model_selection import cross_val_score
        
        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
        
        return cv_results