# Import python libraries
import os, sys
import click
import mlflow
import pandas as pd
import optuna
from prefect import flow, task
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# To import custom libraries
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


# Import custom libraries
from src.etl.utils import read_toml_config, read_parquet_file, dump_pickle

# Define mlflow parameters
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-hyperparameters")


@task(retries=3, retry_delay_seconds=2, name="Splitting data into train and validation")
def create_train_test_split(df: pd.DataFrame, target: str = "price") -> pd.DataFrame:
    X = df.drop(target, axis=1)
    y = df[target]

    # Split data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save training and validation sets
    dest_path = "./src/data/final"
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))

    return X_train, X_val, y_train, y_val


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Run hyperparameter tuning flow")
def run_hyperparameter_tuning(config_path: str):
    # Unpack the configuration file
    config = read_toml_config(config_path)

    # Unpack config
    preprocessed_data_path = config["training"]["processed_data"][
        "preprocessed_train_df"
    ]
    num_trials = config["training"]["hyperparams_settings"]["num_trials"]

    train_df = read_parquet_file(preprocessed_data_path)

    # 1. Get train and test data
    X_train, X_val, y_train, y_val = create_train_test_split(train_df)

    # 2. Optimise hyperparameters with optuna and log using mlflow
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

            # Store the parameters
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # Store the rmse value
            mlflow.log_metric("rmse", rmse)

            return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    run_hyperparameter_tuning()
