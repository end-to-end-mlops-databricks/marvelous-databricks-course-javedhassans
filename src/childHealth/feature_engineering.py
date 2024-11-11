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

class ActigraphFileReader:
    def __init__(self, app_name: str, root_dir: str):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()
        self.dbutils = DBUtils(self.spark)
        self.root_dir = root_dir

    def _read_participant_file(self, file_path: str, participant_id: str) -> Optional[DataFrame]:
        """
        Reads a single participant file and adds the participant ID as a column.
        """
        try:
            data = self.spark.read.parquet(file_path)
            data = data.withColumn("id", F.lit(participant_id))
            print(f"Successfully read file for participant ID: {participant_id}")
            return data
        except Exception as e:
            print(f"Error reading file {file_path} for participant {participant_id}: {e}")
            return None

    def load_all_files(self) -> Optional[DataFrame]:
        """
        Loads all `part-0.parquet` files from participant directories within the root directory.
        """
        all_data: List[DataFrame] = []
        with ThreadPoolExecutor() as executor:
            futures = []

            try:
                participant_dirs = self.dbutils.fs.ls(self.root_dir)
            except Exception as e:
                print(f"Failed to list directory {self.root_dir}: {e}")
                return None

            for dir_info in participant_dirs:
                if not dir_info.path.endswith('/'):
                    continue  # Skip if it's not a directory

                participant_id = dir_info.path.split('=')[-1].strip('/')
                participant_file_path = f"{dir_info.path}part-0.parquet"
                print(f"Looking for file: {participant_file_path}")

                futures.append(executor.submit(self._read_participant_file, participant_file_path, participant_id))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading participant files"):
                participant_data = future.result()
                if participant_data is not None:
                    all_data.append(participant_data)

        if all_data:
            feature_table = all_data[0]
            for df in all_data[1:]:
                feature_table = feature_table.union(df)
            return feature_table
        else:
            print("No data was loaded. Returning an empty DataFrame.")
            return self.spark.createDataFrame([], schema="id STRING")

