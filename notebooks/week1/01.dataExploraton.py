# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install ../mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from childHealth.config import ProjectConfig

warnings.filterwarnings("ignore")

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Define original paths
dirname_train_ts = "/Volumes/mlops_students/javedhassi/data/series_train.parquet"
dirname_test_ts = "/Volumes/mlops_students/javedhassi/data/series_test.parquet"

# Load project configuration from YAML file
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
num_features = config.num_features
cat_features = config.cat_features


def process_file(filename, dirname):
    filepath = os.path.join(dirname, filename, "part-0.parquet")
    df = spark.read.parquet(filepath).drop("step")
    if "id" not in df.columns:
        df = df.withColumn("id", df["relative_date_PCIAT"])
    return df.toPandas(), filename.split("=")[1]


def load_time_series(dirname) -> pd.DataFrame:
    directories = [file.path for file in dbutils.fs.ls(dirname) if file.path.endswith("/")]
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, path.split("/")[-2], dirname): path for path in directories}
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            print(f"Processed {i + 1}/{len(directories)} files")
    stats, indexes = zip(*results, strict=False) if results else ([], [])
    combined_df = pd.concat([df for df in stats], ignore_index=True)
    combined_df["id"] = indexes
    return combined_df


def update(df):
    for c in cat_features:
        df[c] = df[c].fillna("Missing").astype("category")
    return df


# Load time series data
train_ts = load_time_series(dirname_train_ts)
test_ts = load_time_series(dirname_test_ts)

# Load train and test CSV files with Spark
train = spark.read.csv("/Volumes/mlops_students/javedhassi/data/childHealth.csv", header=True, inferSchema=True)
test = spark.read.csv("/Volumes/mlops_students/javedhassi/data/test.csv", header=True, inferSchema=True)

# Convert Spark DataFrames to Pandas DataFrames
train_pd = train.toPandas()
test_pd = test.toPandas()

# Ensure 'id' column exists in both DataFrames
train_pd["id"] = train_pd.get("id", train_pd.index)
test_pd["id"] = test_pd.get("id", test_pd.index)

# Merge the data
train_merged = pd.merge(train_pd, train_ts, how="left", on="id")
test_merged = pd.merge(test_pd, test_ts, how="left", on="id")

# Update the list of numerical features to include time series columns
time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")
num_features += time_series_cols

# Update the train and test DataFrames
train_merged = update(train_merged)
test_merged = update(test_merged)

# Check the updated DataFrames
print(train_merged.head())
print(test_merged.head())

# Read and show the Parquet file
df = spark.read.parquet(
    "/Volumes/mlops_students/javedhassi/data/series_train.parquet/id=00115b9f/part-0.parquet",
    header=True,
    inferSchema=True,
)
df.show()

# Convert to Pandas DataFrame
df_pandas = df.toPandas()

# Filter and show specific train data
train.filter(train.id == "00115b9f").show()
