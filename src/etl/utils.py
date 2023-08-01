import pickle

import pandas as pd
import toml
from loguru import logger


def read_toml_config(path: str) -> dict:
    logger.info("Reading toml config file")
    return toml.load(path)


def read_parquet_file(path: str) -> pd.DataFrame:
    logger.info(f"Reading parquet file from path: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
