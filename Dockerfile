FROM python:3.11.4

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
# Note: torch is installed separately as it's large and may need specific version
# For CPU-only: use torch (default)
# For GPU support: use torch with CUDA, e.g., torch==2.1.0+cu118
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models action_logs

# Set default command
# Note: Main.py contains input() which requires interactive mode
# For non-interactive Docker, consider using environment variables or removing input()
CMD ["python", "Main.py"]