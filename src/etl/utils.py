import pickle

import pandas as pd
import toml


def read_toml_config(path: str) -> dict:
    return toml.load(path)


def read_parquet_file(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
