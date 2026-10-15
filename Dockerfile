FROM python:3.12-slim

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

RUN useradd --create-home app
USER app
EXPOSE 7860
CMD ["uvicorn", "cargokit.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
