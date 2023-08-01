# Docker Deployment
Follow steps to deploy Flask Application

### Step 1: Create Docker Image
Execute the following commands in terminal:

1. docker build -t <APP-NAME> . <br>
2. docker run -p 4545:4545 <APP-NAME>

**Note:** Replace the <APP-NAME> with any suitable name for the app (e.g. flask-app)

### Step 2: Define Inference Data
Define the values of the test example you want to inference on:

**Appropriate selections for categorical variables:** <br>
- **manufacturer** = [ 'ford', 'toyota', 'jeep', 'ram', 'cadillac', 'gmc', 'honda', 'chevrolet', 'dodge', 'lexus', 'jaguar', 'buick', 'chrysler', 'volvo',
'infiniti', 'lincoln', 'acura', 'hyundai', 'mercedes-benz', 'audi', 'bmw', 'mitsubishi', 'nissan', 'subaru', 'alfa-romeo', 'pontiac', 'kia',
'volkswagen', 'fiat', 'rover', 'mazda', 'tesla', 'saturn', 'porsche', 'mini', 'harley-davidson', 'mercury', 'ferrari', 'datsun', 'aston-martin', 'land rover']

- **fuel** = ['gas', 'other', 'diesel', 'hybrid', 'electric']

- **titled_status** = ['clean', 'rebuilt', 'lien', 'salvage', 'missing', 'parts only']

- **transmission** = ['other', 'automatic', 'manual']

- **type** = ['pickup', 'SUV', 'hatchback', 'other', 'mini-van', 'truck', 'sedan', 'coupe', 'offroad', 'van', 'wagon', 'convertible', 'bus']

- **paint_color** = ['white', 'red', 'silver', 'black', 'blue', 'brown', 'grey', 'yellow', 'orange', 'custom', 'green', 'purple']

### Step 3: Run Prediction
Execute the following command in the terminal: <br>
*python deployment/inference_data.py*
