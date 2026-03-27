FROM python:3.11-slim

# Set the working directory
WORKDIR /app

#Install system deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

    # Copy dependencies file and install first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Streamlit config: listen on all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Optional: set OpenAI env variable name (value passed at runtime)
# ENV OPENAI_API_KEY=""

# Expose port
EXPOSE 8501

# Default comman
CMD ["streamlit", "run", "app.py"]