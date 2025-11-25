import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# === RUTAS ===
modelo_path = r"D:\UPAO\CICLO VII\MACHINE LEARNING\ProyectoCarnesV2\modelo_meat_quality_v3.h5"
imagen_path = r"D:\UPAO\CICLO VII\MACHINE LEARNING\ProyectoCarnesV2\imagen_prueba\carne_fresca.jpg"

# === CARGAR MODELO ===
model = tf.keras.models.load_model(modelo_path)
print("✅ Modelo cargado correctamente")

# === DETECTAR TAMAÑO DE ENTRADA ===
input_shape = model.input_shape[1:3]
print(f"📏 Tamaño de entrada del modelo: {input_shape}")

# === CARGAR IMAGEN ===
img = image.load_img(imagen_path, target_size=input_shape)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# === PREDICCIÓN ===
pred = model.predict(x)
clases = ['Fresh', 'Spoiled']

print("\n📈 Valores de predicción crudos:")
print(pred)

if pred.shape[1] == 1:
    # Salida sigmoide → probabilidad de clase 1 (Spoiled)
    prob_spoiled = float(pred[0][0])
    prob_fresh = 1 - prob_spoiled
    etiqueta = clases[1] if prob_spoiled > 0.5 else clases[0]
    prob_final = prob_spoiled if etiqueta == 'Spoiled' else prob_fresh
else:
    # Salida softmax → múltiples clases
    idx = int(np.argmax(pred))
    etiqueta = clases[idx]
    prob_final = float(np.max(pred))

# === RESULTADO ===
print("\n🔍 Resultado de la predicción:")
print(f"➡️ La imagen es probablemente: {etiqueta}")
print(f"📊 Confianza: {prob_final * 100:.4f}%")

# === MOSTRAR IMAGEN ===
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicción: {etiqueta} ({prob_final * 100:.2f}% de confianza)")
plt.show()
