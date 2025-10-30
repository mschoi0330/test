FROM python:3.11-slim
WORKDIR /app

# Optional system deps for OCR if you later add it:
# RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false     STREAMLIT_SERVER_ADDRESS=0.0.0.0     STREAMLIT_SERVER_PORT=8501     PYTHONUNBUFFERED=1

VOLUME ["/app/chroma_db"]
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
