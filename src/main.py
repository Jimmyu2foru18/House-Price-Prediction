import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_processor import DataProcessor
from models.regression_models import RegressionModels
from visualization.visualizer import Visualizer

def main():
    # Initialize components
    data_processor = DataProcessor()
    models = RegressionModels()
    visualizer = Visualizer()

    # Load and preprocess data
    data_path = '../housing.csv'
    if not data_processor.load_data(data_path):
        print("Failed to load data. Please check the file path.")
        return

    # Preprocess data
    X, y = data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data()

    # Train and evaluate models
    model_metrics = {}
    for model_type in ['linear', 'ridge', 'lasso']:
        # Train model
        model = models.train_model(X_train, y_train, model_type)
        
        # Evaluate model
        metrics = models.evaluate_model(model, X_test, y_test)
        model_metrics[model_type] = metrics
        
        print(f"\n{model_type.capitalize()} Regression Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Visualizations
    # Plot feature importance
    feature_importance = data_processor.get_feature_importance()
    importance_plot = visualizer.plot_feature_importance(
        feature_importance,
        feature_importance.index
    )
    visualizer.save_plot(importance_plot, 'feature_importance.png')

    # Plot model comparison
    comparison_plot = visualizer.plot_model_comparison(model_metrics)
    visualizer.save_plot(comparison_plot, 'model_comparison.png')

    # Cross-validation results
    cv_results = models.cross_validate_models(X, y)
    print("\nCross-validation Results:")
    for model_name, scores in cv_results.items():
        print(f"{model_name.capitalize()}:")
        print(f"Mean R² Score: {scores['mean_score']:.4f} (+/- {scores['std_score']:.4f})")

if __name__ == '__main__':
    main()