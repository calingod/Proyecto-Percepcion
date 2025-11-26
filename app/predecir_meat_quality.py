import os
import sys
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Spoiled"]  # 0 = Fresh, 1 = Spoiled


def load_model(model_path: str):
    print(f"Cargando modelo desde: {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_image(model, image_path: str):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prob = model.predict(x)[0][0]  # salida sigmoide
    label_idx = 1 if prob >= 0.9 else 0
    label = CLASS_NAMES[label_idx]

    print(f"\nImagen: {image_path}")
    print(f"Prob( Spoiled ) = {prob:.4f}")
    print(f"Predicción: {label}")
    return label, prob


def main():
    if len(sys.argv) < 2:
        print("Uso: python predecir_meat_quality.py ruta/de/imagen.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = os.environ.get("MODEL_PATH", "modelo_meat_quality_v3.keras")

    model = load_model(model_path)
    predict_image(model, image_path)


if __name__ == "__main__":
    main()