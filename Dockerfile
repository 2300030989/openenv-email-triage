# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Define environment variable
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the app via uvicorn (which handles both FastAPI and Gradio)
CMD ["python", "server/app.py"]
