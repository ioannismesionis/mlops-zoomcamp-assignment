# Import python libraries
import os
import sys
from typing import List

import click
import pandas as pd
from feature_engine.encoding import MeanEncoder
from loguru import logger
from prefect import flow, task

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.utils import dump_pickle, read_parquet_file, read_toml_config


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns, axis=1)


@task(retries=3, retry_delay_seconds=2, name="Encode categorical features")
def encode_categorical_variables(
    df: pd.DataFrame, cat_variables: list[str], target: str = "price"
) -> pd.DataFrame:
    """Encode categorical features using mean encoder.

    Args:
        df (pd.DataFrame): The raw dataframe
        cat_variables (list[str]): Categorical variables to encode
        target (str, optional): The name of the response variable. Defaults to "price".

    Returns:
        pd.DataFrame: Encoded dataframe.
    """
    logger.info("Fitting mean encoder transforming categorical variables")
    encoder = MeanEncoder(variables=cat_variables)

    # Fit and Transform
    encoder.fit(df.drop(target, axis=1), df[target])
    encoded_df = encoder.transform(df.drop(target, axis=1))

    # Save the encoder to a pre-specified path
    encoder_path = "./src/etl/transformers/mean_encoder.pkl"
    logger.info(f"Saving encoder to path: {encoder_path}")
    dump_pickle(encoder, encoder_path)

    return pd.concat([encoded_df, df[target]], axis=1)


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Running Preprocessing Flow")
def run_preprocessing(config_path: str) -> None:
    """Run the preprocessing pipeline step by step.

    Args:
        config_path (str): Path to config

    Returns:
        None
    """
    logger.info("Running the preprocessing pipeline")

    # Read the config file
    config = read_toml_config(config_path)

    # Unpack config file
    drop_cols = config["preprocessing"]["drop_cols"]
    cat_variables = config["preprocessing"]["cat_variables"]
    preprocessed_data_path = config["preprocessing"]["processed_data"][
        "preprocessed_data_path"
    ]

    # Unpack config file for the training pipeline
    train_data_path = config["preprocessing"]["raw_data"]["train_data_path"]

    # Run preprocessing steps
    df = read_parquet_file(train_data_path)
    df = drop_columns(df, drop_cols)
    df = encode_categorical_variables(df, cat_variables)

    # Save the preprocessed training data
    save_path = os.path.join(preprocessed_data_path, "train_df.parquet")
    logger.info(f"Saving preprocessed training data to {save_path}")
    df.to_parquet(save_path, engine="pyarrow")

    logger.info("Preprocessing pipeline completed")


if __name__ == "__main__":
    run_preprocessing()
