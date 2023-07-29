# Import python libraries
import pandas as pd
from datetime import datetime
from prefect import flow, task
import click
import os
import sys
from typing import List
from feature_engine.encoding import MeanEncoder

CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


# Import customer libraries
from src.etl.utils import read_toml_config, read_parquet_file, dump_pickle, load_pickle


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns, axis=1)


@task(retries=3, retry_delay_seconds=2, name="Encode categorical features")
def encode_categorical_variables(
    df: pd.DataFrame,
    cat_variables: list[str],
    target: str = "price",
    fit_encoder: bool = True,
    encoder_path: str = None,
) -> pd.DataFrame:
    if fit_encoder:
        encoder = MeanEncoder(variables=cat_variables)
        encoder.fit(df.drop(target, axis=1), df[target])

        # Save the encoder
        month = datetime.now().month
        day = datetime.now().day
        encoder_path = f"./src/etl/transformers/{month:02d}_{day:02d}_mean_encoder.pkl"
        dump_pickle(encoder, encoder_path)

    # Transform using the encoder
    else:
        # load encoder
        encoder = load_pickle(encoder_path)

    # Transform the data by encoding categorical variables
    encoded_df = encoder.transform(df.drop(target, axis=1))
    df = pd.concat([encoded_df, df[target]], axis=1)

    return df


@click.command()
@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
@flow(name="Running Preprocessing flow")
def run_preprocessing(config_path: str):
    # Unpack the configuration file
    config = read_toml_config(config_path)

    pipeline = config["settings"]["pipeline"]

    dest_path = config["preprocessing"]["processed_data"]["dest_path"]

    if pipeline == "training":
        # Unpack config file for the training pipeline
        train_data_path = config["preprocessing"]["raw_data"]["train_data_path"]
        drop_cols = config["preprocessing"]["drop_cols"]
        cat_variables = config["preprocessing"]["cat_variables"]

        # Read the training data
        df = read_parquet_file(train_data_path)

        df = drop_columns(df, drop_cols)

        # Encode categorical features
        df = encode_categorical_variables(df, cat_variables)

        # Save the training data
        save_path = os.path.join(dest_path, "train_df.parquet")
        df.to_parquet(save_path, engine="pyarrow")

    elif pipeline == "testing":
        # Step 1:
        # Step 2:
        df.to_parquet(os.path.join(dest_path, "test_df.parquet"), engine="pyarrow")
        # dump_pickle(df, os.path.join(dest_path, "test_df.pkl"))

        ...


if __name__ == "__main__":
    run_preprocessing()
