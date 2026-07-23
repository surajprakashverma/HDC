import os

# Reduce TensorFlow logs and force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mnist_digit_secret_key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist.keras")

model = None


def get_model():
    """
    Lazy-load the TensorFlow model only when prediction is requested.
    This helps reduce Render startup memory pressure.
    """
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file 'mnist.keras' not found. Please keep mnist.keras in the project root folder."
            )

        model = load_model(MODEL_PATH)

    return model


def preprocess_image(file):
    """
    Preprocess uploaded digit image for MNIST model prediction.

    Steps:
    1. Convert image to grayscale.
    2. Convert image to binary.
    3. Detect digit boundary.
    4. Crop digit.
    5. Place digit in square canvas.
    6. Resize to 28x28.
    7. Invert image to match MNIST format.
    8. Normalize and reshape.
    """

    img = Image.open(file).convert("L")

    img = np.array(img)

    # Convert to binary image
    # White background = 255, dark digit = 0
    img = np.where(img > 180, 255, 0).astype(np.uint8)

    # Find digit pixels
    coords = np.argwhere(img == 0)

    if len(coords) == 0:
        raise ValueError("No digit detected in image. Please upload a clear handwritten digit.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop digit area
    img = img[y0:y1, x0:x1]

    img = Image.fromarray(img)

    # Make square canvas
    width, height = img.size
    max_dim = max(width, height)

    square = Image.new("L", (max_dim, max_dim), 255)

    square.paste(
        img,
        (
            (max_dim - width) // 2,
            (max_dim - height) // 2
        )
    )

    # Resize to MNIST size
    square = square.resize((28, 28), Image.Resampling.LANCZOS)

    # Invert colors because MNIST uses white digit on black background
    square = ImageOps.invert(square)

    # Normalize
    img = np.array(square).astype("float32") / 255.0

    # CNN model usually expects: batch, height, width, channel
    img = img.reshape(1, 28, 28, 1)

    return img


@app.route("/")
def index():
    result = session.pop("result", None)
    confidence = session.pop("confidence", None)
    error = session.pop("error", None)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "Image" not in request.files:
            session["error"] = "No image uploaded."
            return redirect(url_for("index"))

        file = request.files["Image"]

        if file.filename == "":
            session["error"] = "Please choose an image."
            return redirect(url_for("index"))

        allowed_extensions = {"png", "jpg", "jpeg", "webp"}

        if "." not in file.filename:
            session["error"] = "Invalid file. Please upload a PNG, JPG, JPEG, or WEBP image."
            return redirect(url_for("index"))

        extension = file.filename.rsplit(".", 1)[1].lower()

        if extension not in allowed_extensions:
            session["error"] = "Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP."
            return redirect(url_for("index"))

        img = preprocess_image(file)

        mnist_model = get_model()

        prediction = mnist_model.predict(img, verbose=0)

        pred = int(np.argmax(prediction))
        confidence = round(float(np.max(prediction)) * 100, 2)

        session["result"] = pred
        session["confidence"] = confidence

    except Exception as e:
        session["error"] = str(e)

    return redirect(url_for("index"))


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_PATH)
    }


if __name__ == "__main__":
    app.run(debug=True)
