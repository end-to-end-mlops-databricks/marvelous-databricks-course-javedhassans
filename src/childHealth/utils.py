import pandas as pd
import yaml  # Fixed the import statement
from childHealth.config import ProjectConfig
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(y_test, y_pred):
    """
    Visualize the results of predictions against actual values.
    
    Parameters:
    y_test (array-like): Actual values.
    y_pred (array-like): Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance, feature_names, top_n=10):
    """
    Plot the top N feature importances.
    
    Parameters:
    feature_importance (array-like): Importance scores of features.
    feature_names (list): Names of the features.
    top_n (int): Number of top features to display.
    """
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    
def adjust_predictions(predictions, scale_factor=1.3):
    return predictions * scale_factor