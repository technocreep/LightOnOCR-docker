FROM python:3.10-slim

# Устанавливаем только poppler-utils (для pdf2image) и базовую библиотеку для OpenCV
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем зависимости (убедитесь, что opencv-python-headless в списке)
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    pypdf pdf2image Pillow \
    aiohttp tqdm opencv-python-headless \
    pymorphy2 ru-text-cleaner wordfreq emoji nltk \
    lingua-language-detector \
    symspellpy \ markdown-table-repair

# Копируем код
COPY app.py .

# Запускаем с --reload для мгновенного подхвата изменений в коде
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8609", "--reload"]