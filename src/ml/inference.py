# Import python libraries
import os
import sys

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.utils import read_toml_config, read_parquet_file
from src.etl.preprocessing import drop_columns, encode_categorical_variables


def run_inference(config_path: str) -> None:
    # 1. Read config
    config = read_toml_config(config_path)

    # Unpack configuration file
    drop_cols = config["preprocessing"]["drop_cols"]
    categ_variables = config["preprocessing"]["cat_variables"]
    test_df_path = config["inference"]["test_df_path"]
    cat_encoder_path = config["inference"]["cat_encoder_path"]
    model_path = config["inference"]["model_path"]

    # 2. Read test data
    test_df = read_parquet_file(test_df_path)

    # 3. Drop columns
    test_df = drop_columns(test_df, drop_cols)

    # 4. Encode categorical variables
    test_df = encode_categorical_variables(
        test_df,
        cat_variables=categ_variables,
        fit_encoder=False,
        encoder_path=cat_encoder_path,
    )

    # 5. Load model
    # TODO: Can I load from MLflow??
    y_pred = model.predict(test_df)


if __name__ == "__main__":
    run_inference()
