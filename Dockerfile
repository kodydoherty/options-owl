FROM python:3.12-slim

WORKDIR /app

# LightGBM requires libgomp for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy everything needed for install
COPY pyproject.toml .
COPY options_owl/ options_owl/
COPY scripts/ scripts/

# Install (no cache to keep image small)
RUN pip install --no-cache-dir .

# Persist the SQLite database
VOLUME ["/app/journal"]

CMD ["python", "-m", "options_owl.bot_runner"]
