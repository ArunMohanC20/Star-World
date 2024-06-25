from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static', static_folder='env/static')

# Load the trained model

model = load_model('C:/Users/HP/Desktop/flaskkkk/env/final_constellation_model.h5')

# Define class names
class_names = ['Not a Constellation', 'Orion', 'Scorpio', 'Ursa Major']

# Descriptions for each constellation
descriptions = {
    'Orion': 'Orion is a prominent constellation located on the celestial equator.',
    'Scorpio': 'Scorpio is one of the zodiac constellations, often associated with a scorpion.',
    'Ursa Major': 'Ursa Major, also known as the Great Bear, contains the Big Dipper asterism.'
}

# Function to preprocess and predict image
def predict_image(img_path):
    img = Image.open(img_path).convert('L').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save file to static/uploads inside the env directory
    upload_folder = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Predict the constellation
    prediction_class = predict_image(file_path)
    predicted_class = class_names[prediction_class]

    # Prepare description and example image
    example_image_path = os.path.join(app.static_folder, 'examples', f'{predicted_class.lower()}_example.jpg')
    description = descriptions.get(predicted_class, '')

    uploaded_image = f'/static/uploads/{file.filename}'
    example_image = f'/static/examples/{predicted_class.lower()}_example.jpg'

    # Return JSON response
    return jsonify({
        'uploaded_image': uploaded_image,
        'example_image': example_image,
        'predicted_class': predicted_class,
        'description': description
    })

if __name__ == '__main__':
    app.run(debug=True)
