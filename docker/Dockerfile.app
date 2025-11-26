FROM tensorflow/tensorflow:2.17.0-gpu

# Java para Spark
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/app

# Copiamos requirements y los instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Copiamos el código de la app
COPY . .

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "web_app:app"]
