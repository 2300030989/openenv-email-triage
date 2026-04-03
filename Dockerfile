# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy dependency files first for caching
# We copy them separately to leverage Docker cache layers
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies using uv
# If uv.lock is missing, uv will generate it during sync
RUN uv sync

# Copy the rest of the application code
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Run the app via the entrypoint defined in pyproject.toml
CMD ["python", "server/app.py"]
