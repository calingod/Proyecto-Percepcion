import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Spoiled"]
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_meat_quality_v3_last.keras")

app = Flask(__name__)

print(f"Cargando modelo desde: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

HTML_FORM = """
<!doctype html>
<html lang="es">
  <head>
    <title>Clasificador de Calidad de Carne</title>
    <meta charset="utf-8">
  </head>
  <body>
    <h1>Clasificador de Calidad de Carne</h1>
    <p>Sube una imagen de carne para que el modelo la evalúe como fresca o podrida.</p>
    <form method="POST" enctype="multipart/form-data" action="/">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Evaluar</button>
    </form>

    {% if result %}
      <h2>Resultado</h2>
      <p><b>Predicción:</b> {{ result.label }}</p>
      <p><b>Prob( Podrida ):</b> {{ result.prob }}</p>
    {% endif %}
  </body>
</html>
"""

def preprocess_image(file_storage):
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype="float32")
    # OJO: no dividimos entre 255 aquí si el modelo ya tiene Rescaling(1./255)
    x = np.expand_dims(x, axis=0)
    return x

def predict(x):
    prob = model.predict(x)[0][0]
    label_idx = 1 if prob >= 0.5 else 0
    label = CLASS_NAMES[label_idx]
    return label, float(prob)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template_string(HTML_FORM, result=None)

        file = request.files["file"]
        x = preprocess_image(file)
        label, prob = predict(x)
        result = {
            "label": label,
            "prob": f"{prob:.4f}"
        }

    return render_template_string(HTML_FORM, result=result)

if __name__ == "__main__":
    # Flask escuchando en 0.0.0.0 para que Docker pueda exponerlo
    app.run(host="0.0.0.0", port=5000, debug=False)
