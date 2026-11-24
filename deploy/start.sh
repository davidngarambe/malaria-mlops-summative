#!/bin/sh
# Starts the FastAPI service (internal) and the Streamlit UI (public) in one container.
set -e

uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait for the API before starting the UI
for i in $(seq 1 30); do
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null && break
  sleep 1
done

export API_URL="${API_URL:-http://localhost:8000}"
exec streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true
