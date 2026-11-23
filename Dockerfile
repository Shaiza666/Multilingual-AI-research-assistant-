FROM python:3.12-slim-bookworm

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY . .
EXPOSE 8741
CMD ["python", "-m", "streamlit", "run", "src/mira/app.py", "--server.port", "8741", "--server.address", "0.0.0.0"]
