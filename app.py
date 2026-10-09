
import os
import json
import uuid

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "brinjal_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Model loaded successfully!")
print("Disease classes:", class_names)


# ---------------- HELPER FUNCTIONS ----------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def predict_leaf(image):
    """
    Preprocess an image and predict its class.

    This matches a model that includes its own Rescaling
    layer to convert pixel values from 0-255 to [-1, 1].
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image, dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array, axis=0
    )

    probabilities = model.predict(
        image_array, verbose=0
    )[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model output does not match class_names.json."
        )

    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    return predicted_class, confidence


# ---------------- MAIN PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("leaf_image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a brinjal leaf image."

        elif not allowed_file(uploaded_file.filename):
            error = "Upload a JPG, JPEG, PNG, or WEBP image."

        else:
            try:
                # Validate the uploaded image
                image = Image.open(
                    uploaded_file.stream
                ).convert("RGB")

                prediction, confidence = predict_leaf(image)

                # Create a unique filename
                safe_name = secure_filename(
                    uploaded_file.filename
                )

                extension = safe_name.rsplit(".", 1)[1].lower()

                unique_filename = (
                    f"{uuid.uuid4().hex}.{extension}"
                )

                save_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_filename
                )

                # Save a display copy as JPEG
                image.save(save_path, format="JPEG")

                image_url = (
                    "/static/uploads/" + unique_filename
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):
                error = (
                    "Unable to process this image. "
                    "Please choose a valid leaf photo."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


# ---------------- ERROR HANDLING ----------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        image_url=None,
        error="Image is too large. Maximum size is 8 MB."
    ), 413


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )