from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import json

# Setup paths
BASE_DIR = os.path.abspath(os.path.dirname(file))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'fruit_veg_disease_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Flask app setup
app = Flask(name, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class names
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'Empty file name', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100
    status = "✅ Good to Eat" if "__Healthy" in predicted_class else "❌ Don't Eat"

    # Return simple result page
    return f'''
    <h1>Prediction Result</h1>
    <p><strong>Label:</strong> {predicted_class}</p>
    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
    <p><strong>Status:</strong> {status}</p>
    <img src="/static/uploads/{filename}" alt="Uploaded image" style="max-width:300px;">
    <br><br><a href="/">🔁 Try Another</a>
    '''

if name == 'main':
    app.run(debug=True)