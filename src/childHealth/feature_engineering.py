from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, LongType
import os
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.dbutils import DBUtils

from childHealth.config import ProjectConfig
from databricks.feature_engineering import FeatureFunction, FeatureLookup

import logging
from pyspark.sql.utils import AnalysisException


# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ActigraphFileReader:
    def __init__(self, app_name: str, root_dir: str, catalog_name: str, schema_name: str):
        """
        Initializes the ActigraphFileReader class.
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
            .getOrCreate()
        self.dbutils = DBUtils(self.spark)
        self.root_dir = root_dir
        self.catalog_name = catalog_name
        self.schema_name = schema_name

        # Define the schema for the feature table
        self.schema = StructType([
            StructField("step", LongType(), True),
            StructField("X", FloatType(), True),
            StructField("Y", FloatType(), True),
            StructField("Z", FloatType(), True),
            StructField("enmo", FloatType(), True),
            StructField("anglez", FloatType(), True),
            StructField("non_wear_flag", FloatType(), True),
            StructField("light", FloatType(), True),
            StructField("battery_voltage", FloatType(), True),
            StructField("time_of_day", LongType(), True),
            StructField("weekday", IntegerType(), True),
            StructField("quarter", IntegerType(), True),
            StructField("relative_date_PCIAT", FloatType(), True),
            StructField("id", StringType(), True),
        ])
        logger.info("Initialized ActigraphFileReader with app name %s, root directory %s, catalog %s, and schema %s.",
                    app_name, root_dir, catalog_name, schema_name)

    def _read_participant_file(self, file_path: str, participant_id: str) -> Optional[DataFrame]:
        """
        Reads a single participant file and adds the participant ID as a column.
        """
        try:
            data = self.spark.read.schema(self.schema).parquet(file_path)
            data = data.withColumn("id", F.lit(participant_id))
            logger.info("Successfully read file for participant ID: %s from path: %s", participant_id, file_path)
            return data
        except AnalysisException as e:
            logger.warning("File not found or error reading file %s for participant %s: %s", file_path, participant_id, e)
            return None
        except Exception as e:
            logger.error("Unexpected error reading file %s for participant %s: %s", file_path, participant_id, e)
            return None

    def load_all_files(self) -> Optional[DataFrame]:
        """
        Loads all `part-0.parquet` files from participant directories within the root directory.
        Combines them into a single DataFrame.
        """
        all_data: List[DataFrame] = []
        with ThreadPoolExecutor() as executor:
            futures = []
            try:
                participant_dirs = self.dbutils.fs.ls(self.root_dir)
            except Exception as e:
                logger.error("Failed to list directory %s: %s", self.root_dir, e)
                return None

            for dir_info in participant_dirs:
                if not dir_info.path.endswith('/'):
                    continue  # Skip if it's not a directory

                participant_id = dir_info.path.split('=')[-1].strip('/')
                participant_file_path = f"{dir_info.path}part-0.parquet"
                futures.append(executor.submit(self._read_participant_file, participant_file_path, participant_id))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading participant files"):
                participant_data = future.result()
                if participant_data is not None:
                    all_data.append(participant_data)

        if all_data:
            feature_table = all_data[0]
            for df in all_data[1:]:
                feature_table = feature_table.union(df)
            logger.info("Successfully loaded and combined all participant files.")
            return feature_table
        else:
            logger.warning("No data was loaded from the files. Returning an empty DataFrame.")
            return self.spark.createDataFrame([], schema=self.schema)

    def save_feature_table(self):
        """
        Creates and updates the feature table in Delta format with constraints and change data feed enabled.
        """
        # Ensure auto-merge is enabled for schema changes
        self.spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
        logger.info("Auto-merge enabled for schema changes.")

        # Load all files into a DataFrame
        feature_table = self.load_all_files()

        # Define the table name
        table_name = f"{self.catalog_name}.{self.schema_name}.actigraph_features"
        logger.info("Preparing to create or replace the Delta table: %s", table_name)

        # Step 1: Create or replace the table with the defined schema
        try:
            self.spark.sql(f"""
            CREATE OR REPLACE TABLE {table_name} (
                step BIGINT,
                X FLOAT,
                Y FLOAT,
                Z FLOAT,
                enmo FLOAT,
                anglez FLOAT,
                non_wear_flag FLOAT,
                light FLOAT,
                battery_voltage FLOAT,
                time_of_day BIGINT,
                weekday TINYINT,
                quarter TINYINT,
                relative_date_PCIAT FLOAT,
                id STRING NOT NULL
            )
            USING DELTA
            """)
            logger.info("Table %s created or replaced successfully.", table_name)
        except Exception as e:
            logger.error("Error creating or replacing table %s: %s", table_name, e)
            return

        # Step 2: Add a primary key constraint to the table
        try:
            self.spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT actigraph_pk PRIMARY KEY (id)")
            logger.info("Primary key constraint added to table %s.", table_name)
        except Exception as e:
            logger.error("Error adding primary key constraint to table %s: %s", table_name, e)

        # Step 3: Enable change data feed for the table
        try:
            self.spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
            logger.info("Change data feed enabled for table %s.", table_name)
        except Exception as e:
            logger.error("Error enabling change data feed for table %s: %s", table_name, e)

        # Step 4: Write data to the feature table if there is any
        try:
            if feature_table.count() > 0:
                # Ensure consistent column names and types
                feature_table = feature_table.withColumnRenamed("non-wear-flag", "non_wear_flag") \
                                             .withColumn("weekday", F.col("weekday").cast("TINYINT")) \
                                             .withColumn("quarter", F.col("quarter").cast("TINYINT")) \
                                             .withColumn("id", F.col("id").cast("STRING"))

                # Write data with schema merge enabled
                feature_table.write.format("delta") \
                                   .mode("append") \
                                   .option("mergeSchema", "true") \
                                   .saveAsTable(table_name)
                logger.info("Feature table %s saved and updated successfully.", table_name)
            else:
                logger.info("Feature table is empty; nothing was saved.")
        except Exception as e:
            logger.error("Error writing data to the feature table %s: %s", table_name, e)