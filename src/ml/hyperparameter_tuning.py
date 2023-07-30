# Import python libraries
import os
import sys

import click
import mlflow
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.utils import dump_pickle, read_parquet_file, read_toml_config

# Define mlflow parameters for tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-hyperparameters")


@task(retries=3, retry_delay_seconds=2, name="Splitting into train and validation sets")
def create_train_and_validation_sets(
    df: pd.DataFrame, target: str = "price", save=True
) -> pd.DataFrame:
    X = df.drop(target, axis=1)
    y = df[target]

    # Split data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save training and validation sets
    if save:
        final_data_path = "./src/data/final"
        dump_pickle((X_train, y_train), os.path.join(final_data_path, "train.pkl"))
        dump_pickle((X_val, y_val), os.path.join(final_data_path, "val.pkl"))

    return X_train, X_val, y_train, y_val


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Run Hyperparameter Tuning Flow")
def run_hyperparameter_tuning(config_path: str) -> None:
    """Run hyperparameter tuning

    Args:
        config_path (str): Path to config

    Returns:
        None
    """
    # Read the configuration file
    config = read_toml_config(config_path)

    # Unpack config file
    preprocessed_data_path = config["training"]["processed_data"][
        "preprocessed_train_df"
    ]
    num_trials = config["training"]["hyperparams_settings"]["num_trials"]

    # Read the training data to perform parameter tuning
    train_df = read_parquet_file(preprocessed_data_path)

    # Create training and validation datasets for the hyperparameter tuning
    X_train, X_val, y_train, y_val = create_train_and_validation_sets(train_df)

    # Optimise hyperparameters with optuna and log mlflow results
    def objective(trial):
        with mlflow.start_run():
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
                "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
                "random_state": 42,
                "n_jobs": -1,
            }

            # Store the hyperparameters used
            mlflow.log_params(params)

            # Train random forest and evaluate RMSE in validation dataset
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # Store the rmse value in MLflow
            mlflow.log_metric("rmse", rmse)

            return rmse

    # Start optuna hyperparameter tuning procedure
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    run_hyperparameter_tuning()
