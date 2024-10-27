import pandas as pd
import yml
from childHealth.config import ProjectConfig

def remove_outliers(df: pd.DataFrame, num_features: list, lower_bound: float = 0.25, upper_bound: float = 0.75) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    num_features (list): List of numerical feature names.
    lower_bound (float): Lower quantile bound.
    upper_bound (float): Upper quantile bound.
    
    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    """
    for feature in num_features:
        Q1 = df[feature].quantile(lower_bound)
        Q3 = df[feature].quantile(upper_bound)
        IQR = Q3 - Q1
        df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]
    return df