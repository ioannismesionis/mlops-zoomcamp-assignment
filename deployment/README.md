# Docker Deployment

### Step 1: Create Docker Image


### Step 2: Define Inference Data
manufacturer = ['ford', 'toyota', 'jeep', 'ram', 'cadillac', 'gmc', 'honda',
       'chevrolet', 'dodge', 'lexus', 'jaguar', 'buick', 'chrysler',
       'volvo', 'infiniti', 'lincoln', 'acura', 'hyundai',
       'mercedes-benz', 'audi', 'bmw', 'mitsubishi', 'nissan', 'subaru',
       'alfa-romeo', 'pontiac', 'kia', 'volkswagen', 'fiat', 'rover',
       'mazda', 'tesla', 'saturn', 'porsche', 'mini', 'harley-davidson',
       'mercury', 'ferrari', 'datsun', 'aston-martin', 'land rover']
fuel = ['gas', 'other', 'diesel', 'hybrid', 'electric']
titled_status = ['clean', 'rebuilt', 'lien', 'salvage', 'missing', 'parts only']
transmission = ['other', 'automatic', 'manual']
['pickup', 'SUV', 'hatchback', 'other', 'mini-van', 'truck',
       'sedan', 'coupe', 'offroad', 'van', 'wagon', 'convertible', 'bus']
paint_color = ['white', 'red', 'silver', 'black', 'blue', 'brown', 'grey',
       'yellow', 'orange', 'custom', 'green', 'purple']



### Step 3: Run Prediction
docker build -t flask-app .
docker run -p 5000:5000 flask-app

