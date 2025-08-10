# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs artifacts models plots data

# Expose ports
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "fastapi" ]; then\n\
    exec uvicorn fastapi_app:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "streamlit" ]; then\n\
    exec streamlit run streamlit_dashboard.py --server.address 0.0.0.0 --server.port 8501\n\
elif [ "$1" = "train" ]; then\n\
    exec python main.py\n\
else\n\
    echo "Usage: docker run <image> [fastapi|streamlit|train]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["fastapi"]