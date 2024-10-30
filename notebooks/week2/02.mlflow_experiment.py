# Databricks notebook source
import json
import subprocess
import mlflow

mlflow.set_tracking_uri("databricks")

# COMMAND --------------

mlflow.set_experiment(experiment_name="/Shared/child-health-basic")
mlflow.set_experiment_tags({"repository_name": "child-health"})

# COMMAND ----------
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='child-health'",
)
print(experiments)

# COMMAND ----------
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)
    
# COMMAND ----------
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": '830c17d988742482b639aec763ec731ac2dd4da5',
          "branch": "week1-2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/child-health-basic"],
    filter_string="tags.git_sha='830c17d988742482b639aec763ec731ac2dd4da5'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])

# COMMAND ----------
