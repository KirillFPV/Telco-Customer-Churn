# Этап сборки зависимостей
FROM python:3.11-slim as builder

WORKDIR /app

# Установка системных зависимостей для компиляции
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование файла зависимостей
COPY requirements.txt .

# Создание виртуального окружения и установка зависимостей
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Финальный этап
FROM python:3.11-slim

# Создание не-root пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash -d /app appuser

WORKDIR /app

# Копирование виртуального окружения из builder этапа
COPY --from=builder /opt/venv /opt/venv

# Копирование файлов приложения
COPY app.py data_pipeline.py model.cbm ./

# Установка прав и владельца
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Переключение на не-root пользователя
USER appuser

# Установка переменных окружения
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Открытие порта
EXPOSE 8080

# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]