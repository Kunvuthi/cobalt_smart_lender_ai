# Use the official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements (assumes a separate one for streamlit; adjust if shared)
COPY ../../requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Streamlit app files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "cobalt_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]