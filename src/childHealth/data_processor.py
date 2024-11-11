import pandas as pd
import numpy as np
from childHealth.config import ProjectConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql import SparkSession


class TrainDataProcessor:
    def __init__(self, train_df: pd.DataFrame, config: ProjectConfig):
        self.train_df = train_df
        self.config = config
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target

    def preprocess_data(self):
        """Preprocess the train dataset by handling missing values and data type conversions."""
        self.handle_missing_values()
        self.convert_data_types()
        return self.train_df

    def handle_missing_values(self):
        """Handle missing values in the train dataset."""
        self._fill_numeric_missing_values()
        self._fill_categorical_missing_values()

    def _fill_numeric_missing_values(self):
        """Fill numeric columns with mean."""
        numeric_cols = self.train_df[self.num_features].apply(pd.to_numeric, errors='coerce')
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numeric_cols)
        
        self.train_df[self.num_features] = pd.DataFrame(imputed_data, columns=self.num_features, index=self.train_df.index)

    def _fill_categorical_missing_values(self):
        """Fill categorical columns with mode."""
        for col in self.cat_features:
            self.train_df[col].fillna(self.train_df[col].mode()[0], inplace=True)

    def convert_data_types(self):
        """Convert categorical columns to appropriate data types."""
        self._convert_sex_to_binary()

    def _convert_sex_to_binary(self):
        """Convert 'Sex' to binary encoding if it's part of numerical features."""
        if 'Basic_Demos-Sex' in self.num_features:
            self.train_df['Basic_Demos-Sex'] = self.train_df['Basic_Demos-Sex'].map({'Male': 1, 'Female': 0})

    def feature_engineering(self):
        """Perform feature engineering to create new features."""
        self.add_age_groups()
        self.one_hot_encode_seasons()
        self.calculate_behavioral_scores()
        self.add_interaction_features()
        return self.train_df

    def add_age_groups(self):
        """Add age groups based on age."""
        if 'Basic_Demos-Age' in self.num_features:
            self.train_df['Age_Group'] = pd.cut(self.train_df['Basic_Demos-Age'], bins=[0, 12, 17, 25], labels=['Child', 'Teen', 'Young Adult'])

    def one_hot_encode_seasons(self):
        """One-hot encode season columns."""
        for col in self.cat_features:
            if 'Season' in col:
                one_hot = pd.get_dummies(self.train_df[col], prefix=col)
                self.train_df = pd.concat([self.train_df, one_hot], axis=1)

    def calculate_behavioral_scores(self):
        """Calculate behavioral and psychological indicators."""
        self._bin_pciat_total_score()
        self._categorize_internet_use()

    def _bin_pciat_total_score(self):
        """Bin PCIAT total score."""
        if 'PCIAT-PCIAT_Total' in self.num_features:
            self.train_df['PCIAT_Bin'] = pd.cut(self.train_df['PCIAT-PCIAT_Total'], bins=[0, 20, 40, 60], labels=['Mild', 'Moderate', 'Severe'])

    def _categorize_internet_use(self):
        """Categorize internet use."""
        if 'PreInt_EduHx-computerinternet_hoursday' in self.num_features:
            self.train_df['Internet_Use_Category'] = pd.cut(self.train_df['PreInt_EduHx-computerinternet_hoursday'], bins=[0, 1, 3, 6, np.inf], labels=['Low', 'Moderate', 'High', 'Very High'])

    def add_interaction_features(self):
        """Add interaction features, such as age-adjusted scores."""
        self._add_age_adjusted_cgas()
        self._add_bmi_categories()

    def _add_age_adjusted_cgas(self):
        """Add age-adjusted CGAS Score."""
        if 'CGAS-CGAS_Score' in self.num_features and 'Basic_Demos-Age' in self.num_features:
            self.train_df['Age_Adjusted_CGAS'] = self.train_df['CGAS-CGAS_Score'] / self.train_df['Basic_Demos-Age']

    def _add_bmi_categories(self):
        """Add BMI Categories."""
        if 'Physical-BMI' in self.num_features:
            self.train_df['BMI_Category'] = pd.cut(self.train_df['Physical-BMI'], bins=[0, 18.5, 25, 30, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    def scale_numeric_features(self):
        """Scale numeric features in the final dataset."""
        scaler = StandardScaler()
        self.train_df[self.num_features] = scaler.fit_transform(self.train_df[self.num_features])

    def handle_missing_target(self):
        """Handle missing values in the target column by replacing them with a specific value (e.g., 4.0)."""
        new_value = 4.0  # New value to replace NaNs in the target column
        self.train_df[self.target].fillna(new_value, inplace=True)

    def keep_relevant_columns(self):
        """Keep only relevant columns."""
        relevant_columns = self.cat_features + self.num_features + [self.target, 'id']
        self.train_df = self.train_df[relevant_columns]

    def process(self):
        """Run the complete processing pipeline."""
        self.preprocess_data()
        self.feature_engineering()
        self.scale_numeric_features()
        self.handle_missing_target()
        self.keep_relevant_columns()
        return self.train_df

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the DataFrame (self.train_df) into training and test sets.

        Parameters:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

        Returns:
        tuple: A tuple containing the training and test sets.
        """
        train_set, test_set = train_test_split(self.train_df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """
        Save the train and test sets into Databricks tables.

        Parameters:
        train_set (pd.DataFrame): The training set.
        test_set (pd.DataFrame): The test set.
        spark (SparkSession): The Spark session.
        """
        self._add_timestamp_and_save(train_set, 'train_set', spark)
        self._add_timestamp_and_save(test_set, 'test_set', spark)
        self._enable_change_data_feed(spark)

    def _add_timestamp_and_save(self, df: pd.DataFrame, table_name: str, spark: SparkSession):
        """Add timestamp column and save DataFrame to Databricks table."""
        df_with_timestamp = spark.createDataFrame(df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
        df_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}")

    def _enable_change_data_feed(self, spark: SparkSession):
        """Enable Change Data Feed for train and test sets."""
        for table in ['train_set', 'test_set']:
            spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{table} "
                      "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")