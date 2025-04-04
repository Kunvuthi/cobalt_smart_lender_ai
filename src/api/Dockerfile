# Base image with Python
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements.txt into image
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire src/ directory into container (your app lives inside /app now)
COPY src/api /app

# Expose FastAPI port
EXPOSE 8000

# Default FastAPI run command
CMD ["uvicorn", "app.cobalt_fast_api:app", "--host", "0.0.0.0", "--port", "8000"]