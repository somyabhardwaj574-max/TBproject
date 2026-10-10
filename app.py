import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "tb_model.tflite")
KERAS_PATH = os.path.join(BASE_DIR, "tb_model.keras")

app = Flask(__name__)
IMG_SIZE = (150, 150)

# ── Model Loader ─────────────────────────────────────────
# LiteRT / TFLite uses ~40MB RAM (ideal for Render 512MB limit)
model_mode = None
interpreter = None
keras_model = None
input_idx = None
output_idx = None

if os.path.exists(TFLITE_PATH):
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError:
                from tflite_runtime.interpreter import Interpreter
        
        interpreter = Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]
        model_mode = "tflite"
        print("Loaded lightweight model: tb_model.tflite")
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")

if not model_mode and os.path.exists(KERAS_PATH):
    try:
        from tensorflow.keras.models import load_model
        keras_model = load_model(KERAS_PATH)
        model_mode = "keras"
        print("Loaded Keras model: tb_model.keras")
    except Exception as e:
        print(f"Failed to load Keras model: {e}")

if not model_mode:
    print("Warning: Neither TFLite nor Keras model could be loaded.")


def predict_image(img_array):
    if model_mode == "tflite":
        interpreter.set_tensor(input_idx, img_array)
        interpreter.invoke()
        return float(interpreter.get_tensor(output_idx)[0][0])
    elif model_mode == "keras":
        return float(keras_model.predict(img_array, verbose=0)[0][0])
    else:
        raise RuntimeError("No model is currently loaded")


# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok", mode=model_mode)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify(error="No file uploaded"), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify(error="No file selected"), 400

        # Preprocess directly in memory using PIL
        img = Image.open(file.stream).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        raw = predict_image(img_array)

        tb_percent = round(raw * 100, 2)
        normal_percent = round(100 - tb_percent, 2)

        return jsonify(normal=normal_percent, tb=tb_percent)
    except Exception as e:
        return jsonify(error=f"Prediction error: {str(e)}"), 500


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)