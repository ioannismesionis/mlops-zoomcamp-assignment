# mlops-zoomcamp-assignment
Final Assignment of the MLOps Zoomcamp Course (2023)

# Car Price Prediction Project README

This Python project aims to predict the selling price of cars based on their features. It follows a series of steps to preprocess the data, perform hyperparameter tuning, register the best model, conduct inference on unseen data, and generate monitoring reports.

## Business Problem

Predicting the selling price of used cars is crucial for both buyers and sellers. For buyers, it helps in making informed decisions about the fair value of the vehicle they are interested in. For sellers, it helps in setting the right price to attract potential buyers and maximize profits. This project seeks to build a machine learning model that takes various car features as input and predicts the price at which a car can be sold.

## Solution

The solution involves the following steps for code execution:

### Set up

1. Create a virtual environment and execute "pip install -r requirements.txt" to install the required dependencies.

### Step 1

2. Execute "mlflow ui --backend-store-uri sqlite:///mlflow.db" to start MLflow, which will be used for tracking model runs and parameters.

### Step 2

3. Execute "prefect server start" to start the Prefect server, which will be used to execute and manage Prefect flows.

### Step 3

4. Run "python src/etl/preprocessing.py" to perform data preprocessing.

    a. This step runs the Prefect flow for data preprocessing.
    
    b. It saves the encoder used for preprocessing to "src/etl/transformers/mean_encoder.pkl".
    
    c. It generates the processed training data "train_df.parquet" in "src/data/preprocessed/train_df.parquet".

### Step 4

5. Run "python src/ml/hyperparameter_tuning.py" to perform hyperparameter tuning.

    a. This step runs the Prefect flow for hyperparameter tuning.
    
    b. It stores the runs in MLflow, enabling tracking and comparison of different hyperparameter configurations.
    
    c. It uses optuna for hyperparameter tuning, seeking the best model hyperparameters.
    
    d. It stores the data used for hyperparameter tuning in "src/data/final/".

### Step 5

6. Run "python src/ml/register_best_model.py" to register the best model.

    a. This step runs the Prefect flow for registering the best model.
    
    b. It stores the runs in MLflow and registers the best model in the model registry.
    
    c. It stores the best model in "src/etl/transformers/model.pkl".

### Step 6

7. Run "python src/ml/inference.py" to conduct inference on unseen data.

    a. This step takes unseen data from "src/data/raw/vehicles_2023-05.parquet" and conducts all the necessary preprocessing.
    
    b. It loads the best model and produces inferences for the unseen data.
    
    c. This step runs the Prefect flow of doing inference on the unseen data stored in ...

### Step 7

8. Execute "evidently ui" to generate the workspace folder to store the monitoring web reports.

### Step 8

9. Run "python src/ml/monitoring.py" to generate performance and data draft reports.

    a. These reports can be opened with a web browser for visualization or through the Evidently UI.

## Project Implementations

The project includes the following implementations:

- **black formatter**: Code formatting using black for consistent code style.

- **isort formatter**: Import sorting using isort for organized imports.

- **CI/CD using GitHub Actions**: Continuous Integration and Continuous Deployment are set up using GitHub Actions, ensuring automated testing and deployment.

- **pre-commit yaml**: Pre-commit hooks are set up to enforce code formatting and checks before commits.

- **unit tests**: Unit tests are provided to test the various functions and ensure the correctness of the code.

## Running the Project

To run the project, follow the steps mentioned in the "Code Execution" section. Please ensure you have the necessary dependencies installed in the virtual environment and that you have set up MLflow and Prefect as described.

For monitoring, open Evidently UI to visualize the generated reports.

## Contributing

If you want to contribute to the project or report any issues, please follow the standard guidelines for contributing and issue reporting provided in the repository.

We hope this project helps in predicting accurate car prices and enables better decision-making for both buyers and sellers. Happy predicting!

