# Import python libraries
import os
import sys

from loguru import logger
import pandas as pd
from flask import Flask, request, jsonify

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.utils import load_pickle

# Define Flask application
app = Flask("price-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Flask app that can predict a single testing example

    Returns:
        dict: Price prediction
    """
    # Define entry points for paths
    # _CATEGORICAL_VARIABLES = [
    #     "manufacturer",
    #     "fuel",
    #     "title_status",
    #     "transmission",
    #     "type",
    #     "paint_color",
    # ]
    # raw_car_info = {
    #     "year": 2018,
    #     "odometer": 20856.0,
    #     "manufacturer": "ford",
    #     "fuel": "gas",
    #     "title_status": "clean",
    #     "transmission": "automatic",
    #     "type": "SUV",
    #     "paint_color": "red",
    #     "lat": 32.590000,
    #     "long": -85.480000,
    # }

    _CAT_ENCODER_PATH = "./src/etl/transformers/mean_encoder.pkl"
    _MODEL_RUN_PATH = "./src/etl/transformers/model.pkl"

    # Get testing example
    raw_car_info = request.get_json()
    logger.info("Getting raw car info")
    test_df = pd.DataFrame(raw_car_info, index=[0])

    # Load categorical transformer
    logger.info("Loading categorical transformer")
    encoder = load_pickle(_CAT_ENCODER_PATH)
    test_df = encoder.transform(test_df)

    # Load trained model and predict on testing example
    logger.info("Loading ML model and predict price value")
    model = load_pickle(_MODEL_RUN_PATH)
    price_pred = model.predict(test_df)

    # Get prediction result
    logger.info("Returning result")
    result = {"price": price_pred[0]}

    # return result
    return jsonify(result)


if __name__ == "__main__":
    # predict_endpoint()
    app.run(debug=True, host="0.0.0.0", port=4545)
