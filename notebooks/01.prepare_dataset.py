# Databricks notebook source

from pyspark.sql import SparkSession
from childHealth.config import ProjectConfig
from childHealth.data_processor import DataProcessor
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load project configuration from YAML file
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

# Load the child health dataset into a Pandas DataFrame
df = spark.read.csv(
    "/Volumes/mlops_students/javedhassi/data/childHealth.csv", 
    header=True, 
    inferSchema=True
).toPandas()

# COMMAND ----------

# Initialize DataProcessor with the loaded DataFrame and configuration
data_processor = DataProcessor(pandas_df=df, config=config)

# Preprocess the data
data_processor.preprocess()

# Split the data into training and testing sets
train_set, test_set = data_processor.split_data()

# Save the processed datasets to the catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)