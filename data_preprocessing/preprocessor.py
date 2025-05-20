import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []

    def load_data(self, file_path):
        """Load dataset from CSV file"""
        return pd.read_csv(file_path)

    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Drop rows with any missing values
        df = df.dropna()
        # Remove duplicates
        df = df.drop_duplicates()
        return df

    def feature_engineering(self, df):
        """Create new features and transform existing ones"""
        # Example feature: age of house
        df['house_age'] = pd.datetime.now().year - df['year_built']
        return df

    def prepare_features(self, df):
        """Prepare features for modeling"""
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Identify feature types
        self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Scale numerical features
        X[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        
        # Encode categorical features
        X = pd.get_dummies(X, columns=self.categorical_cols)
        
        return X, y

    def split_data(self, X, y, test_size=0.2):
        """Split dataset into training and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=42)