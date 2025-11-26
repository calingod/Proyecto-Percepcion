import os
import glob
from kafka import KafkaProducer

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "meat_quality_images")


def main():
    data_dir = os.environ.get("DATA_DIR", "/workspace/data_split")
    # Puedes cambiar esto a "test" si quieres usar el set de prueba
    pattern = os.path.join(data_dir, "test", "*", "*.jpg")

    files = glob.glob(pattern)
    if not files:
        print(f"No se encontraron imágenes con patrón: {pattern}")
        return

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: v.encode("utf-8"),
    )

    print(f"Enviando {len(files)} rutas de imagen al tópico '{TOPIC}'...")

    for path in files:
        print(f"[Kafka] Enviando: {path}")
        producer.send(TOPIC, value=path)

    producer.flush()
    producer.close()
    print("Productor terminado.")


if __name__ == "__main__":
    main()
