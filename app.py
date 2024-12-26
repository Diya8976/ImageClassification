import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Path to the saved model
MODEL_PATH = "space_image_classifier.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define class labels (ensure this matches your dataset's class labels)
CLASS_LABELS = ['Planet', 'Star', 'Nebula', 'Spacecraft']  # Update based on your classes

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create uploads folder if it doesn't exist

# Add allowed file extensions check
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        if file:
            # Save uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make Prediction
            predictions = model.predict(img_array)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]

            # Remove the file after prediction (optional)
            os.remove(filepath)

            # Add try-except blocks for model prediction
            try:
                predictions = model.predict(img_array)
            except Exception as e:
                return f"Error processing image: {str(e)}", 500

            return render_template("result.html", predicted_class=predicted_class)

    return render_template("index.html")
    

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
