FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY models ./models
COPY data ./data
COPY deploy/start.sh ./start.sh
RUN chmod +x start.sh

# 8000 = FastAPI, 8501 = Streamlit (Render injects $PORT for the public UI)
EXPOSE 8000 8501
CMD ["./start.sh"]
