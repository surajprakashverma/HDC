import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "mnist_digit_secret_key"

model = load_model("mnist.keras")


def preprocess_image(file):
    img = Image.open(file).convert("L")

    img = np.array(img)

    # Convert to binary
    img = np.where(img > 180, 255, 0).astype(np.uint8)

    # Find digit boundaries
    coords = np.argwhere(img == 0)

    if len(coords) == 0:
        raise ValueError("No digit detected in image.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    img = img[y0:y1, x0:x1]

    img = Image.fromarray(img)

    # Make square canvas
    width, height = img.size
    max_dim = max(width, height)

    square = Image.new("L", (max_dim, max_dim), 255)

    square.paste(
        img,
        ((max_dim - width) // 2,
         (max_dim - height) // 2)
    )

    # Resize to MNIST size
    square = square.resize((28, 28))

    # Invert colors (MNIST: white digit on black background)
    square = ImageOps.invert(square)

    img = np.array(square).astype("float32") / 255.0

    img = img.reshape(1, 28, 28)

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

        img = preprocess_image(file)

        prediction = model.predict(img, verbose=0)

        pred = int(np.argmax(prediction))
        confidence = round(float(np.max(prediction)) * 100, 2)

        session["result"] = pred
        session["confidence"] = confidence

    except Exception as e:
        session["error"] = str(e)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)