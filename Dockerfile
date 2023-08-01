FROM python:3.10.0-slim


# Set the working directory inside the Docker image
WORKDIR /app

# Copy all the files and folders from the local directory to the Docker image
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose the port on which your Flask application will run
EXPOSE 5000

# Set the entry point for the container 
CMD ["python", "./src/deployment/predict.py"]
#ENTRYPOINT [ "python", "./src/ml/inference.py"]
