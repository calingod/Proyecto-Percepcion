import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras

# ========== CONFIGURACIÓN ==========
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_BASE = 20
EPOCHS_FINE = 15

DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data_split")
MODEL_NAME = "modelo_meat_quality_v3_last.keras"

# ========== DATASET ==========
print(f"Usando DATA_DIR = {DATA_DIR}")

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

print("Clases detectadas por Keras:", train_ds.class_names)

# ========== OPTIMIZACIÓN PIPELINE ==========
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ========== DATA AUGMENTATION ==========
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# ========== MODELO BASE ==========
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Congelamos primero

# ========== MODELO COMPLETO ==========
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="meat_quality_cnn")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========== CALLBACKS ==========
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_NAME,
        save_best_only=True,
        monitor="val_accuracy"
    ),
    keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    )
]

# ========== CONFIGURAR MLFLOW ==========
mlflow.set_tracking_uri("file:/workspace/mlruns")
mlflow.set_experiment("meat_quality_experiments")

with mlflow.start_run():
    # Log de hiperparámetros
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs_base", EPOCHS_BASE)
    mlflow.log_param("epochs_fine", EPOCHS_FINE)

    print("\n=== ENTRENAMIENTO BASE ===")
    history_base = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_BASE,
        callbacks=callbacks
    )

    # Log de métricas del último epoch base
    mlflow.log_metric("base_val_accuracy", history_base.history["val_accuracy"][-1])

    print("\n=== FINE-TUNING ===")
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE
    )

    mlflow.log_metric("fine_val_accuracy", history_fine.history["val_accuracy"][-1])

    # Guardado final
    model.save(MODEL_NAME)
    print(f"\nModelo guardado como: {MODEL_NAME}")

    # Registrar el modelo en MLflow
    mlflow.keras.log_model(model, "meat_quality_model")

