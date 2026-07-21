# AI Handwritten Digit Recognition

A Flask-based AI web application that predicts handwritten digits from uploaded images using a trained TensorFlow/Keras MNIST model.

The application allows users to upload an image of a handwritten digit, preprocesses the image into the MNIST format, and predicts the digit from 0 to 9.

---

## Live Features

- Upload handwritten digit image
- Supports PNG, JPG and JPEG files
- Drag and drop image upload
- Image preview before prediction
- TensorFlow/Keras model prediction
- Confidence score display
- Modern glassmorphism UI
- Animated background and loader
- Flask session-based result display
- Ready for Render deployment

---

## Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- NumPy
- Pillow
- HTML5
- CSS3
- JavaScript
- Bootstrap Icons
- Render

---

## Model Details

The model is trained on the MNIST handwritten digit dataset.

### Dataset

MNIST contains grayscale images of handwritten digits.

- Image size: 28 x 28 pixels
- Classes: 10
- Digits: 0 to 9
- Training images: 60,000
- Testing images: 10,000

---

## Input Processing

When a user uploads an image, the Flask app performs the following preprocessing steps:

1. Converts the uploaded image to grayscale
2. Resizes the image to 28 x 28 pixels
3. Converts the image to a NumPy array
4. Inverts the pixel values
5. Normalizes pixel values between 0 and 1
6. Reshapes the image to match model input shape
7. Sends the image to the trained model for prediction

---

## Project Structure

```text
MNIST-Digit-App/
│
├── app.py
├── mnist.pkl
├── requirements.txt
├── render.yaml
├── README.md
│
└── templates/
    └── index.html