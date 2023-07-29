import os, sys

CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

from src.etl.utils import read_toml_config, read_parquet_file, dump_pickle, load_pickle
from src.etl.preprocessing import drop_columns


def run_inference(config_path: str) -> None:
    # 1. Read config
    config = read_toml_config(config_path)

    # Unpack configuration file
    drop_cols = config["preprocessing"]["drop_cols"]
    test_df_path = config["inference"]["test_df_path"]
    cat_encoder_path = config["inference"]["cat_encoder_path"]
    model_path = config["inference"]["model_path"]

    # 2. Read test data
    test_df = read_parquet_file(test_df_path)

    # 3. Drop columns
    test_df = drop_columns(test_df, drop_cols)

    # 4. Load encoder
    cat_encoder = load_pickle(cat_encoder_path)
    test_df = cat_encoder.transform(test_df)

    # 5. Load model
    # TODO: Can I load from MLflow??
    y_pred = model.predict(test_df)

    print("Mean prediction", y_pred.mean())


if __name__ == "__main__":
    run_inference()
