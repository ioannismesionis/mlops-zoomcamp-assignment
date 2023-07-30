# Import python libraries
import os
import sys

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

from src.etl.utils import read_toml_config


# Helper function to create a temporary TOML file with the given content
def create_temp_toml_file(content: str) -> str:
    temp_file_path = "temp_config.toml"
    with open(temp_file_path, "w") as f:
        f.write(content)
    return temp_file_path


def test_read_toml_config():
    # Create some sample TOML content
    toml_content = """
    [database]
    host = "localhost"
    port = 5432
    username = "admin"
    password = "securepassword"
    """

    # Create a temporary TOML file
    temp_file_path = create_temp_toml_file(toml_content)

    try:
        # Call the function under test
        config = read_toml_config(temp_file_path)

        # Assert that the returned value is a dictionary
        assert isinstance(config, dict)

        # Assert that the content matches the expected dictionary
        expected_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "admin",
                "password": "securepassword",
            }
        }
        assert config == expected_config

    finally:
        # Clean up the temporary TOML file
        os.remove(temp_file_path)
