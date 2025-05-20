import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        self.X = None
        self.y = None

    def load_data(self, file_path):
        """Load the Boston Housing dataset from CSV file."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def preprocess_data(self):
        """Preprocess the data including handling missing values and scaling."""
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")

        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Separate features and target
        self.y = self.data['MEDV']
        self.X = self.data.drop('MEDV', axis=1)

        # Scale features
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns
        )

        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Please preprocess data first.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def get_feature_importance(self):
        """Calculate feature correlation with target variable."""
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")

        correlations = self.data.corr()['MEDV'].sort_values(ascending=False)
        return correlations