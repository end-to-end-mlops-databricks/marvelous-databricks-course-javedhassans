import pandas as pd
from childHealth.config import ProjectConfig
from childHealth.utils import remove_outliers
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql import SparkSession
import yaml

class DataProcessor:
    """
    Class to handle data processing tasks such as splitting data and saving to Databricks catalog.
    """
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        """
        Initialize the DataProcessor with a DataFrame and configuration settings.
        
        Parameters:
        pandas_df (pd.DataFrame): The DataFrame to process.
        config (ProjectConfig): Configuration settings.
        """
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """
        Perform data preprocessing tasks including handling missing values, outliers, 
        and data type conversions.
        
        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """

        # Convert numerical features to numeric type
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove outliers in numerical features
        lower_bound = self.config.parameters['lower_bound']
        upper_bound = self.config.parameters['upper_bound']
        self.df = remove_outliers(self.df, num_features, lower_bound, upper_bound)
        
        # Impute missing values for numerical features with the median
        for col in num_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Convert categorical features to category type
        cat_features = self.config.cat_features
        for col in cat_features:
            self.df[col] = self.df[col].astype('category')
        
        # Impute missing values for categorical features with the most frequent value
        for col in cat_features:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        # Handle missing values in the target column by replacing them with a specific value (e.g., 4.0)
        target = self.config.target
        new_value = 4.0  # New value to replace NaNs in the target column
        self.df[target].fillna(new_value, inplace=True)
        
        # Keep only relevant columns
        relevant_columns = cat_features + num_features + [target, 'Id']
        self.df = self.df[relevant_columns]
        
        return self.df

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the DataFrame (self.df) into training and test sets.
        
        Parameters:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
        Returns:
        tuple: A tuple containing the training and test sets.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """
        Save the train and test sets into Databricks tables.
        
        Parameters:
        train_set (pd.DataFrame): The training set.
        test_set (pd.DataFrame): The test set.
        spark (SparkSession): The Spark session.
        """
        # Add timestamp column to train and test sets
        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        # Save train set to Databricks table
        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        
        # Save test set to Databricks table
        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

        # Enable Change Data Feed for train and test sets
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
                  "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
                  "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
