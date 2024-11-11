import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyspark.sql import DataFrame

from childHealth.config import ProjectConfig
from databricks import feature_engineering
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ActigraphAggregation") \
    .getOrCreate()
    
# Initialize DBUtils
dbutils = DBUtils(spark)


class ActigraphAggregation:
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.config = config
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name

    def load_data(self, participant_id):
        file_path = f"{self.root_dir}/id={participant_id}/part-0.parquet"
        if not dbutils.fs.ls(file_path):
            print(f"File not found for participant {participant_id}")
            return spark.createDataFrame([], schema="id STRING")  # Return an empty Spark DataFrame
        data = spark.read.parquet(file_path)
        data = data.withColumn("id", F.lit(participant_id))
        return data

    def aggregate_actigraphy(self, data):
        aggregated_df = data.groupBy("id").agg(
            F.mean("X").alias("X_mean"), F.stddev("X").alias("X_std"), F.max("X").alias("X_max"), F.min("X").alias("X_min"),
            F.mean("Y").alias("Y_mean"), F.stddev("Y").alias("Y_std"), F.max("Y").alias("Y_max"), F.min("Y").alias("Y_min"),
            F.mean("Z").alias("Z_mean"), F.stddev("Z").alias("Z_std"), F.max("Z").alias("Z_max"), F.min("Z").alias("Z_min"),
            F.mean("enmo").alias("enmo_mean"), F.stddev("enmo").alias("enmo_std"), F.max("enmo").alias("enmo_max"), F.min("enmo").alias("enmo_min"),
            F.mean("anglez").alias("anglez_mean"),
            F.sum("non-wear_flag").alias("non_wear_flag_sum"),
            F.mean("light").alias("light_mean"), F.stddev("light").alias("light_std"), F.max("light").alias("light_max"), F.min("light").alias("light_min"),
            F.mean("battery_voltage").alias("battery_voltage_mean")
        )
        return aggregated_df

    def temporal_aggregations(self, data):
        data = data.withColumn("weekday_flag", F.when(F.col("weekday") < 5, "weekday").otherwise("weekend"))
        data = data.withColumn(
            "time_period",
            F.when(F.col("time_of_day") < 6 * 3600, "night")
             .when((F.col("time_of_day") >= 6 * 3600) & (F.col("time_of_day") < 12 * 3600), "morning")
             .when((F.col("time_of_day") >= 12 * 3600) & (F.col("time_of_day") < 18 * 3600), "afternoon")
             .otherwise("evening")
        )
        
        temporal_agg = data.groupBy("id", "weekday_flag", "time_period").agg(
            F.mean("enmo").alias("enmo_mean"),
            F.mean("light").alias("light_mean"),
            F.sum("non-wear_flag").alias("non_wear_flag_sum")
        ).groupBy("id").pivot("weekday_flag_time_period").agg(
            F.first("enmo_mean"), F.first("light_mean"), F.first("non_wear_flag_sum")
        )
        return temporal_agg

    def activity_ratios(self, data):
        total_time = data.groupBy("id").count().alias("total_time")
        non_wear_time = data.groupBy("id").agg(F.sum("non-wear_flag").alias("non_wear_time"))
        
        ratios = total_time.join(non_wear_time, "id", "left_outer").withColumn(
            "non_wear_ratio", F.col("non_wear_time") / F.col("total_time")
        )
        return ratios.select("id", "non_wear_ratio")

    def process_participant_data(self, participant_id):
        data = self.load_data(participant_id)
        if data.rdd.isEmpty():
            return None
        
        aggregate_data = self.aggregate_actigraphy(data)
        temporal_data = self.temporal_aggregations(data)
        ratio_data = self.activity_ratios(data)

        participant_data = aggregate_data.join(temporal_data, "id", "left").join(ratio_data, "id", "left")
        return participant_data



    def process_all_participants(self) -> DataFrame:
        all_data = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for id_folder in dbutils.fs.ls(self.root_dir):
                if not id_folder.name.startswith('id='):
                    continue
                participant_id = id_folder.name.split('=')[-1]
                futures.append(executor.submit(self.process_participant_data, participant_id))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing participants"):
                participant_data = future.result()
                if participant_data is not None and not participant_data.rdd.isEmpty():
                    all_data.append(participant_data)

        # Check if all_data is empty
        if not all_data:
            print("No participant data was loaded. Returning an empty DataFrame.")
            return spark.createDataFrame([], schema="id STRING")

        # Concatenate all Spark DataFrames using union
        feature_table = all_data[0]
        for df in all_data[1:]:
            feature_table = feature_table.union(df)

        return feature_table


    def save_to_spark_table(self, feature_table):
        feature_table.createOrReplaceTempView("temp_feature_table")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.catalog_name}.{self.schema_name}.actigraph_features AS
            SELECT * FROM temp_feature_table
        """)
        spark.sql(f"ALTER TABLE {self.catalog_name}.{self.schema_name}.actigraph_features SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# # Initialize with the root directory containing participant data
# aggregator = ActigraphAggregation(root_dir="dbfs:/Volumes/mlops_students/javedhassi/data/series_train.parquet/", config=config)

# # Process all participants and get the final aggregated feature table
# feature_table = aggregator.process_all_participants()

# # Save the feature table to Spark
# aggregator.save_to_spark_table(feature_table)

# # Inspect the feature table
# feature_table.show()
