import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (224, 224)
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data_split")
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_meat_quality_v3_last.keras")

def main():
    val_dir = os.path.join(DATA_DIR, "val")
    print("Usando carpeta de validación:", val_dir)
    print("Cargando modelo:", MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="binary",
        shuffle=False
    )

    class_names = val_ds.class_names
    print("Clases detectadas:", class_names)

    y_true = []
    y_pred = []

    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch)
        preds = (probs >= 0.5).astype("int32")  # umbral 0.5

        y_true.extend(y_batch.numpy().astype("int32").flatten())
        y_pred.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
