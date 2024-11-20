import logging
import os

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from childHealth.config import ProjectConfig

# Load configuration
config = ProjectConfig.from_yaml("../../project_config.yml")

# Create logs directory if it doesn't exist
log_file = config.logging.file
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # This will also print logs to the console
    ],
)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
            numeric_cols = self.train_df[self.num_features].apply(pd.to_numeric, errors="coerce")
            imputer = SimpleImputer(strategy="mean")

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
        """Convert 'Sex' to binary encoding if it's part of numerical features and not already binary encoded."""
        if "Basic_Demos-Sex" in self.cat_features:
            # Map the values to binary encoding
            self.train_df["Basic_Demos-Sex"] = self.train_df["Basic_Demos-Sex"].map({1.0: "Male", 0.0: "Female"})
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

    def add_age_groups(self):
        """Add age groups based on age."""
        logging.info("Adding age groups")
        if "Basic_Demos-Age" in self.num_features:
            self.train_df["Age_Group"] = pd.cut(
                self.train_df["Basic_Demos-Age"], bins=[0, 12, 17, 25], labels=["Child", "Teen", "Young Adult"]
            )
            logging.info("Age groups added")

    def one_hot_encode_seasons(self):
        """One-hot encode season columns."""
        logging.info("One-hot encoding seasons")
        for col in self.cat_features:
            if "Season" in col:
                one_hot = pd.get_dummies(self.train_df[col], prefix=col)
                self.train_df = pd.concat([self.train_df, one_hot], axis=1)
                logging.info(f"One-hot encoded {col}")

    def calculate_behavioral_scores(self):
        """Calculate behavioral and psychological indicators."""
        logging.info("Calculating behavioral scores")
        # Bin PCIAT total score
        if "PCIAT-PCIAT_Total" in self.num_features:
            self.train_df["PCIAT_Bin"] = pd.cut(
                self.train_df["PCIAT-PCIAT_Total"], bins=[0, 20, 40, 60], labels=["Mild", "Moderate", "Severe"]
            )
            logging.info("PCIAT total score binned")

        # Categorize internet use
        if "PreInt_EduHx-computerinternet_hoursday" in self.num_features:
            self.train_df["Internet_Use_Category"] = pd.cut(
                self.train_df["PreInt_EduHx-computerinternet_hoursday"],
                bins=[0, 1, 3, 6, np.inf],
                labels=["Low", "Moderate", "High", "Very High"],
            )
            logging.info("Internet use categorized")

    def add_interaction_features(self):
        """Add interaction features, such as age-adjusted scores."""
        logging.info("Adding interaction features")
        # Age-adjusted CGAS Score
        if "CGAS-CGAS_Score" in self.num_features and "Basic_Demos-Age" in self.num_features:
            self.train_df["Age_Adjusted_CGAS"] = self.train_df["CGAS-CGAS_Score"] / self.train_df["Basic_Demos-Age"]
            logging.info("Age-adjusted CGAS score added")

        # BMI Categories
        if "Physical-BMI" in self.num_features:
            self.train_df["BMI_Category"] = pd.cut(
                self.train_df["Physical-BMI"],
                bins=[0, 18.5, 25, 30, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
            )
            logging.info("BMI categories added")

    def scale_numeric_features(self):
        """Scale numeric features in the final dataset."""
        logging.info("Scaling numeric features")
        scaler = StandardScaler()
        self.train_df[self.num_features] = scaler.fit_transform(self.train_df[self.num_features])
        logging.info("Numeric features scaled")

    def handle_missing_target(self):
        """Handle missing values in the target column by replacing them with a specific value (e.g., 4.0)."""
        logging.info("Handling missing target values")
        new_value = 4.0  # New value to replace NaNs in the target column
        self.train_df[self.target].fillna(new_value, inplace=True)
        logging.info("Missing target values handled")

    def keep_relevant_columns(self):
        """Keep only relevant columns."""
        logging.info("Keeping relevant columns")
        relevant_columns = self.num_features + self.cat_features + [self.target]
        self.train_df = self.train_df[relevant_columns]
        logging.info("Relevant columns kept")

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
        self._add_timestamp_and_save(train_set, "train_set", spark)
        self._add_timestamp_and_save(test_set, "test_set", spark)
        self._enable_change_data_feed(spark)
        logging.info("Datasets saved to catalog and change data feed enabled")

    def _add_timestamp_and_save(self, df: pd.DataFrame, table_name: str, spark: SparkSession):
        """Add timestamp and save DataFrame to catalog."""
        logging.info(f"Adding timestamp and saving {table_name}")
        spark_df = spark.createDataFrame(df)
        spark_df = spark_df.withColumn("timestamp", to_utc_timestamp(current_timestamp(), "UTC"))
        spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        logging.info(f"{table_name} saved with timestamp")

    def _enable_change_data_feed(self, spark: SparkSession):
        """Enable change data feed for the catalog."""
        logging.info("Enabling change data feed")
        spark.sql("ALTER TABLE train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        spark.sql("ALTER TABLE test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        logging.info("Change data feed enabled")
