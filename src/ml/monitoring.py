# Import python libraries
import os
import sys
import click
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.utils import read_parquet_file, read_toml_config
from src.ml.inference import get_best_model


@click.option(
    "--config_path",
    default="./src/config/config.toml",
    help="Path to config for orchestration",
)
def run_monitoring(config_path: str) -> None:
    reference_data_path = config["monitoring"]["reference_data_path"]
    current_data_path = config["monitoring"]["current_data_path"]
    model_run_path = config["inference"]["model_run_path"]
    categorical_features = config["preprocessing"]["cat_variables"]

    # 1. Read config
    config = read_toml_config(config_path)

    # 2. Read reference and current data
    reference_df = read_parquet_file(reference_data_path)
    current_df = read_parquet_file(current_data_path)

    # 3. Load model
    model = get_best_model(model_run_path)

    # 4. Predict on current and reference data
    current_pred = model.predict(current_df)
    reference_pred = model.predict(reference_df)

    current_df["prediction"] = current_pred
    reference_df["prediction"] = reference_pred

    # Set evidently column mapping
    column_mapping = ColumnMapping()

    column_mapping.target = "price"
    column_mapping.prediction = "prediction"
    column_mapping.categorical_features = categorical_features

    # Create regression report of model performance
    regression_report = Report(metrics=[RegressionPreset()])
    regression_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # Create data drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    print("Saving...")
    regression_report.save_html("test_suite.html")
    drift_report.save_html("test_drift_report.html")


if __name__ == " __main__":
    run_monitoring()
