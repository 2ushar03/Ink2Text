FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    make \
    libopenblas-dev \
    liblapack-dev \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/

RUN pip install --upgrade pip

RUN pip install flask>=2.1.2 \
    flask-cors==3.0.10 \
    pillow==9.5.0 \
    pytesseract==0.3.9 \
    pyspellchecker==0.7.0 \
    nltk==3.8.1 \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    paddlepaddle==2.5.0 \
    opencv-python-headless==4.7.0.72 \
    paddleocr==2.6.1.3

RUN python -c "from paddleocr import PaddleOCR; PaddleOCR()"

COPY ./app.py /app/
COPY ./ocr_utils.py /app/

EXPOSE 5000

CMD ["python", "app.py"]
