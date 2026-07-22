import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

app.secret_key = "minst_digit_secret_key"

model = pickle.load(open("mnist.pkl", "rb"))

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

        img = Image.open(file).convert("L")
        img = img.resize((28, 28))

        img = np.array(img)
        img = 255 - img
        img = img / 255

        # Change this depending on your model
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img)

        pred = np.argmax(prediction, axis=1)[0]
        confidence = round(float(np.max(prediction)) * 100, 2)

        session["result"] = int(pred)
        session["confidence"] = confidence

    except Exception as e:
        session["error"] = str(e)

    return redirect(url_for("index"))
if __name__=="__main__":

    app.run(debug=True)