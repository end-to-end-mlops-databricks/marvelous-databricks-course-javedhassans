# Databricks notebook source

%pip install childhealth_mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from childHealth.config import ProjectConfig
from childHealth.feature_engineering import ActigraphAggregation
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks import feature_engineering


import warnings
warnings.filterwarnings("ignore")

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()



# COMMAND ----------

# Load project configuration from YAML file
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")


# COMMAND ----------

# create feature table for artigraph tables

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()


# COMMAND ----------
aggregator = ActigraphAggregation(
                     root_dir = "/Volumes/mlops_students/javedhassi/data/series_test.parquet/",
                     config = config)




# COMMAND ----------
feature_table = aggregator.process_all_participants()
# COMMAND ----------
