# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY req-app.txt req.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r req.txt

# Copy the rest of the application code into the container
COPY app/main.py main.py

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run your app
# We use 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]