import os
import sys
from preprocessing.data_processor import DataProcessor
from models.regression_models import RegressionModels
from visualization.visualizer import Visualizer
import matplotlib.pyplot as plt
import pandas as pd

def run_pipeline():
    print("\n===== House Price Prediction Pipeline =====\n")
    
    # Initialize components
    data_processor = DataProcessor()
    models = RegressionModels()
    visualizer = Visualizer()
    
    # Create output directory for plots
    os.makedirs('output', exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_path = 'housing.csv'
    if not data_processor.load_data(data_path):
        print("Failed to load data. Please check the file path.")
        return
    
    X, y = data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data()
    print(f"Data loaded and preprocessed. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Step 2: Train and evaluate models
    print("\nStep 2: Training and evaluating models...")
    model_metrics = {}
    predictions = {}
    
    for model_type in ['linear', 'ridge', 'lasso']:
        print(f"\nTraining {model_type.capitalize()} Regression model...")
        
        # Train model
        model = models.train_model(X_train, y_train, model_type)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[model_type] = y_pred
        
        # Evaluate model
        metrics = models.evaluate_model(model, X_test, y_test)
        model_metrics[model_type] = metrics
        
        print(f"{model_type.capitalize()} Regression Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Step 3: Create visualizations
    print("\nStep 3: Creating visualizations...")
    
    # Plot correlation matrix
    print("Generating correlation matrix...")
    corr_matrix = visualizer.plot_correlation_matrix(data_processor.data)
    visualizer.save_plot(corr_matrix, 'output/correlation_matrix.png')
    
    # Plot feature importance
    print("Generating feature importance plot...")
    feature_importance = data_processor.get_feature_importance()
    importance_plot = visualizer.plot_feature_importance(
        feature_importance,
        feature_importance.index
    )
    visualizer.save_plot(importance_plot, 'output/feature_importance.png')
    
    # Plot model comparison
    print("Generating model comparison plot...")
    comparison_plot = visualizer.plot_model_comparison(model_metrics)
    visualizer.save_plot(comparison_plot, 'output/model_comparison.png')
    
    # Plot predictions vs actual for best model
    best_model = max(model_metrics, key=lambda x: model_metrics[x]['r2'])
    print(f"\nGenerating predictions plot for best model ({best_model})...")
    pred_plot = visualizer.plot_predictions_vs_actual(y_test, predictions[best_model])
    visualizer.save_plot(pred_plot, f'output/{best_model}_predictions.png')
    
    # Plot residuals for best model
    print(f"Generating residuals plot for best model ({best_model})...")
    residuals_plot = visualizer.plot_residuals(y_test, predictions[best_model])
    visualizer.save_plot(residuals_plot, f'output/{best_model}_residuals.png')
    
    print("\n===== Pipeline completed successfully =====")
    print(f"All visualizations saved to the 'output' directory.")
    print(f"Best performing model: {best_model.capitalize()} Regression (R² = {model_metrics[best_model]['r2']:.4f})")

if __name__ == '__main__':
    run_pipeline()