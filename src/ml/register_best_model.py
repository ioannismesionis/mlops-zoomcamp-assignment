# Import python libraries
import os
import sys
from typing import Dict

import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper function
from src.etl.utils import dump_pickle, load_pickle, read_toml_config

# Define mlflow tracking parameters
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")


@task(retries=3, retry_delay_seconds=2, name="Train best random forest")
def train_and_log_model(data_path: str, params: Dict) -> None:
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        for param in params:
            params[param] = int(params[param])

        # Store the parameters
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        best_rmse = 1_000_000_000_000_000  # Arbitrary large number to compare with RMSE
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_rf = rf

        # Log the random forest regressor model
        mlflow.sklearn.log_model(rf, artifact_path="mlflow_models")

    # Save best random forest regressor locally
    dump_pickle(best_rf, "./src/etl/transformers/model.pkl")


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Register Best ML Model")
def run_register_model(config_path: str) -> None:
    """Register the best model after hyperparameter tuning."

    Args:
        config_path (str): Path to config

    Returns:
        None
    """
    _EXPERIMENT_NAME = "random-forest-best-model"
    mlflow.set_experiment(_EXPERIMENT_NAME)

    # Read the configuration file
    config = read_toml_config(config_path)

    # Unpack config file
    HYPERPARAMS_EXPERIMENT_NAME = config["mlflow"]["hyperparams_experiment_name"]
    MAX_RESULTS = config["mlflow"]["max_results"]
    final_data_path = config["training"]["processed_data"]["final_data_path"]

    # Get the all the runs from the hyperparameter experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(HYPERPARAMS_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=MAX_RESULTS,  # Number of top models that need to be evaluated to decide which one to promote
        order_by=["metrics.rmse ASC"],  # Order from best (top 1) to worst run
    )

    # From all active runs, train and log the ML model performance
    for run in runs:
        train_and_log_model(data_path=final_data_path, params=run.data.params)

    # Select the model with the lowest test RMSE (i.e. best model)
    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # Register the best model in model registry of MLflow
    best_model_uri = "runs:/{}/model".format(best_run.info.run_id)
    mlflow.register_model(model_uri=best_model_uri, name="best-rf-regressor-model")


if __name__ == "__main__":
    run_register_model()
