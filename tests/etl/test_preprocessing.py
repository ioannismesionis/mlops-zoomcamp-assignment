# Import python libraries
import os
import sys

import pandas as pd

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import function to be tested
from src.etl.preprocessing import drop_columns


def test_drop_columns():
    # Create a sample DataFrame
    data = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
    }
    df = pd.DataFrame(data)

    # Define the columns to drop
    columns_to_drop = ["A", "C"]

    # Call the function under test
    result_df = drop_columns(df, columns_to_drop)

    # Assert that the returned value is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Assert that the dropped columns are not present in the result DataFrame
    assert "A" not in result_df.columns
    assert "C" not in result_df.columns

    # Assert that the remaining columns are the same as the original DataFrame
    assert result_df.columns.tolist() == ["B"]

    # Assert that the data in the remaining column is unchanged
    assert result_df["B"].tolist() == [4, 5, 6]

    # Assert that the original DataFrame is not modified (immutable function)
    assert df.columns.tolist() == ["A", "B", "C"]

    # Test with an empty list of columns
    result_df_empty = drop_columns(df, [])
    assert result_df_empty.equals(
        df
    )  # The result should be equal to the original DataFrame
