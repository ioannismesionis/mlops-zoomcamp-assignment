# Import python libraries
import os
import sys

import click
import pandas as pd
from flask import Flask, request, jsonify

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions
from src.etl.preprocessing import encode_categorical_variables
from src.ml.inference import get_best_model


# Define Flask application
app = Flask("price-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    _CATEGORICAL_VARIABLES = [
        "manufacturer",
        "fuel",
        "title_status",
        "transmission",
        "type",
        "paint_color",
    ]

    _CAT_ENCODER_PATH = "./src/etl/transformers/mean_encoder.pkl"
    _MODEL_RUN_PATH = "./src/etl/transformers/"

    raw_car_info = request.get_json()
    pred_df = pd.DataFrame(raw_car_info, index=[0])
    pred_df["price"] = -1  # Placeholder for price

    pred_df = encode_categorical_variables.fn(
        pred_df,
        cat_variables=_CATEGORICAL_VARIABLES,
        fit_encoder=False,
        encoder_path=_CAT_ENCODER_PATH,
    )

    model = get_best_model.fn(_MODEL_RUN_PATH)
    price_pred = model.predict(pred_df.drop("price", axis=1))

    result = {"price": price_pred[0]}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4545)
