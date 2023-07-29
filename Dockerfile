FROM python:3.10.0-slim


# Set the working directory inside the Docker image
WORKDIR /app

# Copy all the files and folders from the local directory to the Docker image
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set the entry point for the container (replace 'app.py' with your application's main Python script)
# CMD ["python", "./src/ml/inference.py"]
ENTRYPOINT [ "python", "./src/ml/inference.py"]
