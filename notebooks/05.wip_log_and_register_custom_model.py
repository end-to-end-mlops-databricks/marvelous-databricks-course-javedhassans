# Databricks notebook source
# MAGIC %pip install childhealth_mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from childHealth.data_processor import ProjectConfig
from childHealth.utils import adjust_predictions

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()


# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
model_parameters = config.model_parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/child-health"],
    filter_string="tags.branch='week1-2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# COMMAND ----------


class ChildHealthModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------

wrapped_model = ChildHealthModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/child-health-pyfunc")
git_sha = "830c17d988742482b639aec763ec731ac2dd4da5"

with mlflow.start_run(tags={"branch": "week1-2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    dataset = mlflow.data.from_spark(train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/mlops_with_databricks-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-child-health-model",
        code_paths=["mlops_with_databricks-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-child-health-model")
loaded_model.unwrap_python_model()
# COMMAND ----------