# Databricks notebook source

#MAGIC %pip install childhealth_mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall


# COMMAND ----------

#MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from childHealth.config import ProjectConfig
from childHealth.feature_engineering import ActigraphFileReader
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# Load project configuration from YAML file
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")


# COMMAND ----------
Actigraph = ActigraphFileReader(
    app_name = "ActigraphAggregation",
                     root_dir = "/Volumes/mlops_students/javedhassi/data/series_train.parquet/",
                     catalog_name = config.catalog_name,
                     schema_name = config.schema_name)
# COMMAND ----------
Actigraph.save_feature_table()
# COMMAND ----------
