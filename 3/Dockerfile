# Используем официальный образ Python
FROM python

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir \
    optuna \
    scikit-learn \
    psycopg2-binary \
    numpy \
    pandas

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем все файлы из локальной директории в контейнер
COPY . .

# Указываем команду по умолчанию при запуске контейнера
#CMD ["python", "app.py"]
