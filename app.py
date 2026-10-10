import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tb_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model once at startup
model = load_model(MODEL_PATH)
IMG_SIZE = (150, 150)


# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No file selected"), 400

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Direct model sigmoid prediction
    raw = float(model.predict(img_array)[0][0])  # sigmoid output (0 to 1)

    tb_percent = round(raw * 100, 2)
    normal_percent = round(100 - tb_percent, 2)

    return jsonify(normal=normal_percent, tb=tb_percent)


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)