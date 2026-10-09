

import os

# Use CPU instead of GPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

# ==========================================
# 1. APP CONFIGURATION
# ==========================================

app = Flask(__name__)

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


# ==========================================
# 2. LOAD TRAINED MODEL AND CLASS NAMES
# ==========================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not os.path.isfile(CLASS_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError(
        "class_names.json must contain a non-empty JSON list."
    )

print("Brinjal Plant AI model loaded successfully.")
print("Available classes:", class_names)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def predict_leaf(image):
    """
    Predict the class of a brinjal leaf image.

    This preprocessing expects the trained model to contain
    its own Rescaling layer, as in the earlier training code.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "The number of model output classes does not match "
            "the classes in class_names.json."
        )

    predicted_index = int(np.argmax(probabilities))

    prediction = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    return prediction, confidence


# ==========================================
# 4. HOMEPAGE AND IMAGE PREDICTION
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("leaf_image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select a brinjal leaf image."

        elif not allowed_file(uploaded_file.filename):
            error = (
                "Invalid file type. Please upload "
                "JPG, JPEG, PNG, or WEBP."
            )

        else:
            try:
                # Read and validate the image
                image = Image.open(
                    uploaded_file.stream
                ).convert("RGB")

                # Predict disease class
                prediction, confidence = predict_leaf(image)

                # Save a unique image for display
                filename = f"{uuid.uuid4().hex}.jpg"

                save_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                image.save(save_path, format="JPEG")

                image_url = f"/static/uploads/{filename}"

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):
                prediction = None
                confidence = None
                image_url = None

                error = (
                    "Could not process this image. "
                    "Please upload a valid leaf photograph."
                )

            except Exception:
                app.logger.exception("Prediction failed.")

                prediction = None
                confidence = None
                image_url = None

                error = (
                    "Prediction failed. Please try another image "
                    "or check the application logs."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


# ==========================================
# 5. FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        image_url=None,
        error="The image is too large. Maximum upload size is 8 MB."
    ), 413


# ==========================================
# 6. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
