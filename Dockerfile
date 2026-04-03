# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy dependency files and package metadata first
COPY pyproject.toml uv.loc[k] README.md ./

# Install dependencies using uv.
# We skip the lockfile check if it's not present to ensure the build continues.
RUN if [ -f uv.lock ]; then \
        uv sync --frozen; \
    else \
        uv sync; \
    fi

# Copy the rest of the application code
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Run the app via the entrypoint defined in pyproject.toml
CMD ["python", "server/app.py"]
