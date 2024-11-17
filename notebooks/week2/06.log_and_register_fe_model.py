# Databricks notebook source

# Install necessary packages
#MAGIC %pip install childhealth_mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------
# Restart the Python environment
#MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Import libraries
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import DataFrame, functions as F
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from childHealth.config import ProjectConfig

# COMMAND ----------
# Initialize Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set up MLflow tracking URIs
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

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
# Load training and test datasets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
actigraph_table = spark.table(f"{catalog_name}.{schema_name}.actigraph_features")
actigraph_table = actigraph_table.drop("non_wear_flag")

# COMMAND ----------
# Define function to aggregate actigraphy data
def aggregate_actigraphy(data: DataFrame) -> DataFrame:
    """
    Aggregate actigraphy data for each participant with summary statistics.
    """
    aggregated_df = data.groupBy("id").agg(
        F.mean("X").alias("X_mean"),
        F.stddev("X").alias("X_std"),
        F.max("X").alias("X_max"),
        F.min("X").alias("X_min"),
        
        F.mean("Y").alias("Y_mean"),
        F.stddev("Y").alias("Y_std"),
        F.max("Y").alias("Y_max"),
        F.min("Y").alias("Y_min"),
        
        F.mean("Z").alias("Z_mean"),
        F.stddev("Z").alias("Z_std"),
        F.max("Z").alias("Z_max"),
        F.min("Z").alias("Z_min"),
        
        F.mean("enmo").alias("enmo_mean"),
        F.stddev("enmo").alias("enmo_std"),
        F.max("enmo").alias("enmo_max"),
        F.min("enmo").alias("enmo_min"),
        
        F.mean("light").alias("light_mean"),
        F.stddev("light").alias("light_std"),
        F.max("light").alias("light_max"),
        F.min("light").alias("light_min"),
        
        F.mean("battery_voltage").alias("battery_voltage_mean")
    )
    return aggregated_df

# Aggregate actigraphy data
aggregated_features_df = aggregate_actigraphy(actigraph_table)

# Register the aggregated DataFrame as a temporary view for SQL insertion
aggregated_features_df.createOrReplaceTempView("aggregated_features_view")

# COMMAND ----------
# Create a feature table for actigraphy aggregated features
spark.sql(f"""
    CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.actigraph_aggregated_features
    ( id STRING NOT NULL,
        X_mean DOUBLE,
        X_std DOUBLE,
        X_max DOUBLE,
        X_min DOUBLE,
        Y_mean DOUBLE,
        Y_std DOUBLE,
        Y_max DOUBLE,
        Y_min DOUBLE,
        Z_mean DOUBLE,
        Z_std DOUBLE,
        Z_max DOUBLE,
        Z_min DOUBLE,
        enmo_mean DOUBLE,
        enmo_std DOUBLE,
        enmo_max DOUBLE,
        enmo_min DOUBLE,
        light_mean DOUBLE,
        light_std DOUBLE,
        light_max DOUBLE,
        light_min DOUBLE,
        battery_voltage_mean DOUBLE
    )
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.actigraph_aggregated_features "
          "ADD CONSTRAINT actigraoh_pk PRIMARY KEY(id);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.actigraph_aggregated_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data from the temporary view into the feature table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.actigraph_aggregated_features
    SELECT * FROM aggregated_features_view
""")

# COMMAND ----------
# Define a function to calculate the moving average of battery voltage

function_name = f"{catalog_name}.{schema_name}.moving_average_battery_voltage"
# Define feature table and setup feature engineering
feature_table_name = f"{catalog_name}.{schema_name}.actigraph_aggregated_features"

# Define a function to calculate the house's age using the current year and YearBuilt
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(battery_voltage_mean Double)
RETURNS Double
LANGUAGE PYTHON AS
$$
if battery_voltage_mean is None:
    return None
else:
    return battery_voltage_mean * 1.2

$$
""")

# COMMAND ----------


# Create the training set with feature lookups and feature function for moving average
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["X_mean", "Y_mean", "Z_mean", "battery_voltage_mean"],  # Specify relevant features here
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="battery_voltage_mean_moving_avg",
            input_bindings={"battery_voltage_mean": "battery_voltage_mean"}
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# COMMAND ----------
# Create the test set with feature lookups and feature function for moving average
testing_set = fe.create_training_set(
    df=test_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["X_mean", "Y_mean", "Z_mean", "battery_voltage_mean"],  # Specify relevant features here
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="battery_voltage_mean_moving_avg",
            input_bindings={"battery_voltage_mean": "battery_voltage_mean"}
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------
# Load feature-engineered DataFrame
testing_df = testing_set.load_df().toPandas()

# COMMAND ----------

# Split features and target
X_train = training_df[num_features + cat_features + ["battery_voltage_mean_moving_avg"]]
y_train = training_df[target]
X_test = testing_df[num_features + cat_features + ["battery_voltage_mean_moving_avg"]]
y_test = testing_df[target]

# COMMAND ----------

# Define preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and Random Forest Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**random_forest_parameters))
])

# COMMAND ----------

# Define the MLflow experiment and Git SHA for tracking
mlflow.set_experiment(experiment_name='/Shared/child-health-fe')
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

    # log model with Featue Engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="RandomForestClasf-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
    
mlflow.register_model(
    model_uri=f'runs:/{run_id}/RandomForestClasf-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.child_health_model_random_forest_fe")