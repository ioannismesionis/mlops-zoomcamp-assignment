# Import python libraries
import os
import sys

import click
from loguru import logger
from prefect import flow, task
from sklearn.metrics import mean_squared_error


# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.preprocessing import drop_columns
from src.etl.utils import load_pickle, read_parquet_file, read_toml_config


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
def run_inference(config_path: str) -> None:
    """Run inference on unseen data."

    Args:
        config_path (str): Path to config

    Returns:
        None
    """
    logger.info("Running the inference on unseen data procedure")

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
    y_true = test_df["price"]

    # 4. Encode categorical variables
    encoder = load_pickle(cat_encoder_path)
    test_df = encoder.transform(test_df.drop("price", axis=1))

    # Save the preprocessed test data
    logger.info("Save preprocessed test data")
    save_path = os.path.join(preprocessed_data_path, "test_df.parquet")
    test_df.to_parquet(save_path, engine="pyarrow")

    # 5. Load model
    model = load_pickle(model_run_path)
    y_pred = model.predict(test_df)

    rmse = mean_squared_error(y_true, y_pred, squared=True)
    logger.info(f"Root Mean Square Error (Unseen Data): {rmse}")
    logger.info("Finished the inference on unseen data procedure")


if __name__ == "__main__":
    run_inference()
