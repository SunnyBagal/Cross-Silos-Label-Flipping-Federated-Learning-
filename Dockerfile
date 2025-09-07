# Dockerfile
FROM python:3.10-slim

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy project
COPY . /app

# Install system deps for many Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flower port
EXPOSE 8082

# Default command runs a client; server overwrites in docker-compose
CMD ["python", "client.py"]
