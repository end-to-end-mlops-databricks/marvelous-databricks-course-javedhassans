# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from house_price.data_processor import ProjectConfig
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from house_price.utils import adjust_predictions

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()