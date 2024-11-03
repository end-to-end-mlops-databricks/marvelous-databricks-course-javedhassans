import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyspark.sql import SparkSession
from childHealth.data_processor import ProjectConfig

import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient

from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

class ActigraphAggregation:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
    def load_data(self, participant_id):
        file_path = os.path.join(self.root_dir, f"id={participant_id}", "part-0.parquet")
        if not os.path.exists(file_path):
            print(f"File not found for participant {participant_id}")
            return pd.DataFrame()  # Return an empty DataFrame if file doesn't exist
        data = pd.read_parquet(file_path)
        data['id'] = participant_id
        return data

    def aggregate_actigraphy(self, data):
        aggregated_df = data.groupby('id').agg({
            'X': ['mean', 'std', 'max', 'min'],
            'Y': ['mean', 'std', 'max', 'min'],
            'Z': ['mean', 'std', 'max', 'min'],
            'enmo': ['mean', 'std', 'max', 'min'],
            'anglez': 'mean',
            'non-wear_flag': 'sum',
            'light': ['mean', 'std', 'max', 'min'],
            'battery_voltage': 'mean',
        }).reset_index()
        
        aggregated_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated_df.columns.values]
        return aggregated_df

    def temporal_aggregations(self, data):
        data['weekday_flag'] = data['weekday'].apply(lambda x: 'weekday' if x < 5 else 'weekend')

        conditions = [
            (data['time_of_day'] < 6 * 3600),
            (data['time_of_day'] >= 6 * 3600) & (data['time_of_day'] < 12 * 3600),
            (data['time_of_day'] >= 12 * 3600) & (data['time_of_day'] < 18 * 3600),
            (data['time_of_day'] >= 18 * 3600)
        ]
        choices = ['night', 'morning', 'afternoon', 'evening']
        data['time_period'] = np.select(conditions, choices, default='unknown')
        
        temporal_agg = data.groupby(['id', 'weekday_flag', 'time_period']).agg({
            'enmo': 'mean',
            'light': 'mean',
            'non-wear_flag': 'sum'
        }).unstack(fill_value=0)
        temporal_agg.columns = ['_'.join(col).strip() for col in temporal_agg.columns.values]
        
        return temporal_agg.reset_index()

    def activity_ratios(self, data):
        total_time = data.groupby('id').size().rename('total_time')
        non_wear_time = data.groupby('id')['non-wear_flag'].sum().rename('non_wear_time')
        
        ratios = pd.concat([total_time, non_wear_time], axis=1)
        ratios['non_wear_ratio'] = ratios['non_wear_time'] / ratios['total_time']
        
        return ratios[['non_wear_ratio']].reset_index()
        
    def process_participant_data(self, participant_id):
        data = self.load_data(participant_id)
        if data.empty:
            return pd.DataFrame()
        
        aggregate_data = self.aggregate_actigraphy(data)
        temporal_data = self.temporal_aggregations(data)
        ratio_data = self.activity_ratios(data)

        participant_data = aggregate_data.merge(temporal_data, on='id', how='left')
        participant_data = participant_data.merge(ratio_data, on='id', how='left')
        return participant_data

    def process_all_participants(self):
        all_data = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for id_folder in os.listdir(self.root_dir):
                if not id_folder.startswith('id='):
                    continue
                participant_id = id_folder.split('=')[-1]
                futures.append(executor.submit(self.process_participant_data, participant_id))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing participants"):
                participant_data = future.result()
                if not participant_data.empty:
                    all_data.append(participant_data)

        feature_table = pd.concat(all_data, ignore_index=True)
        return feature_table

    def save_to_spark_table(self, feature_table):
        spark_df = spark.createDataFrame(feature_table)
        spark_df.createOrReplaceTempView("temp_feature_table")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.actigraph_features (
                id STRING,
                X_mean DOUBLE,
                X_std DOUBLE,
                X_max DOUBLE,
                X_min DOUBLE,
                Y_mean DOUBLE,
                Y_std DOUBLE,
                Y_max DOUBLE,
                Y_min DOUBLE,
                Z_mean DOUBLE,
                Z_std DOUBLE,
                Z_max DOUBLE,
                Z_min DOUBLE,
                enmo_mean DOUBLE,
                enmo_std DOUBLE,
                enmo_max DOUBLE,
                enmo_min DOUBLE,
                anglez_mean DOUBLE,
                non_wear_flag_sum DOUBLE,
                light_mean DOUBLE,
                light_std DOUBLE,
                light_max DOUBLE,
                light_min DOUBLE,
                battery_voltage_mean DOUBLE,
                weekday_flag_weekday_morning_mean DOUBLE,
                weekday_flag_weekday_afternoon_mean DOUBLE,
                weekday_flag_weekday_evening_mean DOUBLE,
                weekday_flag_weekday_night_mean DOUBLE,
                weekday_flag_weekend_morning_mean DOUBLE,
                weekday_flag_weekend_afternoon_mean DOUBLE,
                weekday_flag_weekend_evening_mean DOUBLE,
                weekday_flag_weekend_night_mean DOUBLE,
                non_wear_ratio DOUBLE
            )
        """)

        spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.actigraph_features ADD CONSTRAINT actigraph_pk PRIMARY KEY(id)")
        spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.actigraph_features SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

        spark.sql(f"""
            INSERT INTO {catalog_name}.{schema_name}.actigraph_features
            SELECT * FROM temp_feature_table
        """)

# # Initialize with the root directory containing participant data
# aggregator = ActigraphAggregation(root_dir="../../data/series_train.parquet/")

# # Process all participants and get the final aggregated feature table
# feature_table = aggregator.process_all_participants()

# # Save the feature table to Spark
# aggregator.save_to_spark_table(feature_table)

# # Inspect the feature table
# print(feature_table.head())