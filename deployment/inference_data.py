import requests

# Define any options here (look at README for appropriate selections)
car_details = {
    "year": 2018,
    "odometer": 20856.0,
    "manufacturer": "ford",
    "fuel": "gas",
    "title_status": "clean",
    "transmission": "automatic",
    "type": "SUV",
    "paint_color": "red",
    "lat": 32.590000,
    "long": -85.480000,
}

URL = "http://localhost:5000/predict"
response = requests.post(URL, json=car_details)
print(response.json())
