# Use the official Python 3.12 image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project directory to the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir numpy pandas scikit-learn wandb matplotlib

# Command to run when the container starts
CMD ["python", "distance_classification.py"]

