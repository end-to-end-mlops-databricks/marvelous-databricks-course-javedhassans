# Databricks notebook source
%pip install ../mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from childHealth.config import ProjectConfig
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml('../../project_config.yml')

# COMMAND ----------




# COMMAND ----------
# Define the process_file function
def process_file(filename, dirname):
    # Read the Parquet file into a Spark DataFrame
    df = spark.read.parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    
    # Drop the 'step' column
    df = df.drop('step')
    
    # Compute basic statistics
    stats = df.describe().toPandas().values.reshape(-1)
    
    # Extract the filename part
    file_part = filename.split('=')[1]
    
    return stats, file_part

# COMMAND ----------
# Define the load_time_series function
def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df

# COMMAND ----------
# Example usage of the process_file function
dirname = "/Volumes/mlops_students/javedhassi/data/series_test.parquet/"
filename = "part-0.parquet"
stats, file_part = process_file(filename, dirname)
print(stats)
print(file_part)

# COMMAND ----------