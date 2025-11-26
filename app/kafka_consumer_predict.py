import os
from kafka import KafkaConsumer
from predecir_meat_quality import load_model, predict_image

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "meat_quality_images")


def main():
    model_path = os.environ.get("MODEL_PATH", "modelo_meat_quality_v3.keras")
    model = load_model(model_path)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda m: m.decode("utf-8"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="meat_quality_group",
    )

    print(f"Escuchando tópico '{TOPIC}' en {KAFKA_BROKER}...")

    for msg in consumer:
        image_path = msg.value
        print(f"\n[Kafka] Mensaje recibido: {image_path}")
        try:
            predict_image(model, image_path)
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")


if __name__ == "__main__":
    main()
