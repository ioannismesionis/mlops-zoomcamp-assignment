# Import python libraries
import os
import sys
import warnings

# Silent warnings
warnings.filterwarnings("ignore")

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


import pandas as pd

from src.ml.hyperparameter_tuning import create_train_and_validation_sets


# Helper function to create a sample DataFrame
def create_sample_dataframe():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "price": [100, 200, 300, 400, 500],
    }
    return pd.DataFrame(data)


def test_create_train_and_validation_sets():
    # Create a sample DataFrame
    df = create_sample_dataframe()

    # Call the function under test
    X_train, X_val, y_train, y_val = create_train_and_validation_sets.fn(
        df, target="price", save=False
    )

    # Assert that the returned values are DataFrames/Series
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)

    # Assert that the training and validation sets are not empty
    assert not X_train.empty
    assert not X_val.empty
    assert not y_train.empty
    assert not y_val.empty

    # Assert that the features and target columns are correct
    assert set(X_train.columns) == {"feature1", "feature2"}
    assert set(X_val.columns) == {"feature1", "feature2"}
    assert y_train.name == "price"
    assert y_val.name == "price"

    # Assert that the size of the training and validation sets are correct
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # Assert that the training and validation sets are not the same
    assert not X_train.equals(X_val)
    assert not y_train.equals(y_val)

    # Assert that the data is split correctly (e.g., 80% training, 20% validation)
    assert abs(len(X_train) / len(df) - 0.8) < 0.01
    assert abs(len(X_val) / len(df) - 0.2) < 0.01

    # Assert that the pickle files are saved correctly
    assert os.path.exists("./src/data/final/train.pkl")
    assert os.path.exists("./src/data/final/val.pkl")
