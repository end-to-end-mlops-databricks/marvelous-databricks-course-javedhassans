import pandas as pd
import numpy as np
from childHealth.config import ProjectConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql import SparkSession

import logging


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainDataProcessor:
    def __init__(self, train_df: pd.DataFrame, config: ProjectConfig):
        self.train_df = train_df
        self.config = config
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        logging.info("Initialized TrainDataProcessor with config and dataset")

    def preprocess_data(self):
        """Preprocess the train dataset by handling missing values and data type conversions."""
        logging.info("Starting data preprocessing")
        self.handle_missing_values()
        self.convert_data_types()
        logging.info("Data preprocessing completed")
        return self.train_df

    def handle_missing_values(self):
        """Handle missing values in the train dataset."""
        logging.info("Handling missing values")
        self._fill_numeric_missing_values()
        self._fill_categorical_missing_values()

    def _fill_numeric_missing_values(self):
        """Fill numeric columns with mean."""
        try:
            numeric_cols = self.train_df[self.num_features].apply(pd.to_numeric, errors='coerce')
            imputer = SimpleImputer(strategy='mean')

            # Identify and add any missing columns
            missing_columns = set(self.num_features) - set(numeric_cols.columns)
            if missing_columns:
                for col in missing_columns:
                    numeric_cols[col] = np.nan
                    logging.warning(f"Column {col} missing from numeric_cols, filled with NaN")

            # Ensure the columns in numeric_cols match the order in self.num_features
            numeric_cols = numeric_cols[self.num_features]

            # Impute and create a DataFrame
            imputed_data = imputer.fit_transform(numeric_cols)
            imputed_df = pd.DataFrame(imputed_data, columns=self.num_features, index=self.train_df.index)
            self.train_df[self.num_features] = imputed_df
            logging.info("Numeric missing values filled")

        except KeyError as e:
            logging.error(f"Missing columns: {missing_columns}")
            raise e
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            raise e

    def _fill_categorical_missing_values(self):
        """Fill categorical columns with mode."""
        logging.info("Filling categorical missing values")
        for col in self.cat_features:
            self.train_df[col].fillna(self.train_df[col].mode()[0], inplace=True)
            logging.info(f"Filled missing values in {col} with mode")

    def convert_data_types(self):
        """Convert categorical columns to appropriate data types."""
        logging.info("Converting data types")
        self._convert_sex_to_binary()

    def _convert_sex_to_binary(self):
        """Convert 'Sex' to binary encoding if it's part of numerical features."""
        if 'Basic_Demos-Sex' in self.num_features:
            self.train_df['Basic_Demos-Sex'] = self.train_df['Basic_Demos-Sex'].map({'Male': 1, 'Female': 0})
            logging.info("Converted 'Basic_Demos-Sex' to binary")

    def feature_engineering(self):
        """Perform feature engineering to create new features."""
        logging.info("Starting feature engineering")
        self.add_age_groups()
        self.one_hot_encode_seasons()
        self.calculate_behavioral_scores()
        self.add_interaction_features()
        logging.info("Feature engineering completed")
        return self.train_df

    # Rest of the methods...

    def process(self):
        """Run the complete processing pipeline."""
        logging.info("Starting full processing pipeline")
        self.preprocess_data()
        self.feature_engineering()
        self.scale_numeric_features()
        self.handle_missing_target()
        self.keep_relevant_columns()
        logging.info("Processing pipeline completed")
        return self.train_df

    def split_data(self, test_size=0.2, random_state=42):
        logging.info("Splitting data into training and test sets")
        train_set, test_set = train_test_split(self.train_df, test_size=test_size, random_state=random_state)
        logging.info("Data split completed")
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        logging.info("Saving datasets to catalog")
        self._add_timestamp_and_save(train_set, 'train_set', spark)
        self._add_timestamp_and_save(test_set, 'test_set', spark)
        self._enable_change_data_feed(spark)
        logging.info("Datasets saved to catalog and change data feed enabled")