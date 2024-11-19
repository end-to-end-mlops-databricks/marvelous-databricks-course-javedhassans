# Databricks notebook source


import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from pyspark.sql import SparkSession

from childHealth.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

# Load configuration from ProjectConfig
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
random_forest_parameters = config.random_forest_parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.child_health_preds"
online_table_name = f"{catalog_name}.{schema_name}.child_health_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.child_health_model_randomforest/2")

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[["id"]]
preds_df["predicted_sii"] = pipeline.predict(df[cat_features + num_features])

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, primary_keys=["id"], df=preds_df, description="Child health predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table

spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# COMMAND ----------
