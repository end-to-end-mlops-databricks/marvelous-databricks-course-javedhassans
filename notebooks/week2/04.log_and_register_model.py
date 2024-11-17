# Databricks notebook source

# MAGIC %pip install childhealth_mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------
# MAGIC dbutils.library.restartPython()

# Databricks notebook source
# Import necessary libraries
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from childHealth.config import ProjectConfig

# Set up MLflow tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # For registering models to Unity Catalog


# COMMAND ----------
# Load your custom configuration and extract necessary settings
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
random_forest_parameters = config.random_forest_parameters

# Spark session setup
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]


# COMMAND ----------
# Define preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and Random Forest Classifier
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**random_forest_parameters))]
)


# COMMAND ----------
# Define the MLflow experiment and Git SHA for tracking
mlflow.set_experiment(experiment_name="/Shared/child-health")
git_sha = "830c17d988742482b639aec763ec731ac2dd4da5"

# Start an MLflow run to track the training process
with mlflow.start_run(tags={"git_sha": git_sha, "branch": "week1-2"}) as run:
    run_id = run.info.run_id

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance with classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Random Forest with preprocessing")
    mlflow.log_params(random_forest_parameters)
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the dataset source and model in MLflow
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="randomforest-pipeline-model", signature=signature)

# COMMAND ----------
# Register the model in MLflow
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/randomforest-pipeline-model",
    name=f"{catalog_name}.{schema_name}.child_health_model_randomforest",
    tags={"git_sha": git_sha},
)

# COMMAND ----------
# Optionally, load the dataset source (for tracking purposes)
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
