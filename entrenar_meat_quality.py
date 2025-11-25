# ===============================================================
# ENTRENAMIENTO CNN - DETECCIÓN DE CALIDAD DE CARNE (FRESH/SPOILED)
# ===============================================================

import os
import splitfolders  # pip install split-folders
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===============================================================
# RUTAS
# ===============================================================
base_dir = r"D:\UPAO\CICLO VII\MACHINE LEARNING\ProyectoCarnesV2"
data_dir = os.path.join(base_dir, "data")
split_dir = os.path.join(base_dir, "data_split")

# ===============================================================
# 1️⃣ DIVIDIR AUTOMÁTICAMENTE EL DATASET (70/20/10)
# ===============================================================
if not os.path.exists(split_dir):
    splitfolders.ratio(data_dir, output=split_dir, seed=42, ratio=(0.7, 0.2, 0.1))
    print("✅ Dataset dividido correctamente en train/val/test\n")

# ===============================================================
# 2️⃣ GENERADORES DE IMÁGENES
# ===============================================================
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    os.path.join(split_dir, "train"),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    os.path.join(split_dir, "val"),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_gen = datagen.flow_from_directory(
    os.path.join(split_dir, "test"),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"\n📁 Train: {train_gen.samples}")
print(f"📁 Validation: {val_gen.samples}")
print(f"📁 Test: {test_gen.samples}\n")

# ===============================================================
# 3️⃣ MODELO CNN
# ===============================================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ===============================================================
# 4️⃣ ENTRENAMIENTO
# ===============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1
)

# ===============================================================
# 5️⃣ EVALUACIÓN
# ===============================================================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")

# ===============================================================
# 6️⃣ PREDICCIONES Y MÉTRICAS
# ===============================================================
y_pred = model.predict(test_gen)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\n📊 Reporte de Clasificación:\n")
print(classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))

# MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_gen.class_indices.keys()),
            yticklabels=list(test_gen.class_indices.keys()))
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# CURVAS DE ENTRENAMIENTO
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ===============================================================
# 7️⃣ GUARDAR MODELO
# ===============================================================
model.save(os.path.join(base_dir, "modelo_meat_quality_v3.h5"))
print("\n💾 Modelo guardado como 'modelo_meat_quality_v3.h5'")
