# Import python libraries
import os
import sys
import click
import pandas as pd
import mlflow
from prefect import flow, task

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.preprocessing import drop_columns, encode_categorical_variables
from src.etl.utils import read_parquet_file, read_toml_config, load_pickle


@task(retries=3, retry_delay_seconds=2, name="Get best MLflow model")
def get_best_model(model_run_path: str) -> object:
    """Get the best registered model from mlflow.

    Args:
        model_run_path (str): Run path to the model.

    Returns:
        object: sklearn model that can predict.
    """
    # return mlflow.pyfunc.load_model(model_run_path)
    return load_pickle(os.path.join(model_run_path, "model.pkl"))


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Running inference on unseen data")
def run_inference(config_path: str) -> None:
    # 1. Read config
    config = read_toml_config(config_path)

    # Unpack configuration file
    drop_cols = config["preprocessing"]["drop_cols"]
    categorical_variables = config["preprocessing"]["cat_variables"]
    test_df_path = config["inference"]["test_df_path"]
    cat_encoder_path = config["inference"]["cat_encoder_path"]
    model_run_path = config["inference"]["model_run_path"]
    preprocessed_data_path = config["preprocessing"]["processed_data"][
        "preprocessed_data_path"
    ]

    # 2. Read test data
    test_df = read_parquet_file(test_df_path)

    # 3. Drop columns
    test_df = drop_columns(test_df, drop_cols)
    target = test_df["price"]

    # 4. Encode categorical variables
    test_df = encode_categorical_variables(
        test_df,
        cat_variables=categorical_variables,
        fit_encoder=False,
        encoder_path=cat_encoder_path,
    )

    # Save the preprocessed test data
    save_path = os.path.join(preprocessed_data_path, "test_df.parquet")
    test_df.to_parquet(save_path, engine="pyarrow")

    # 5. Load model
    model = get_best_model(model_run_path)
    y_pred = model.predict(test_df.drop("price", axis=1))

    print("Mean predictions:", y_pred.mean())


if __name__ == "__main__":
    run_inference()
