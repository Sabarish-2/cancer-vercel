from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os

# pip install Flask opencv-python-headless numpy tensorflow gdown

app = Flask(__name__)

model_path = 'model/tumor.h5'  # LOCAL

# Download the model file from Google Drive
if not os.path.exists(model_path):
    model_path = 'tumor.h5'  # Update the filename here
    print("Downloading model file...")
    model_url = 'https://drive.google.com/uc?id=13Z0fHGPi4XdQEua8SdyAry1azdMUV8wk'
    gdown.download(model_url, model_path, quiet=False)
    print("Model file downloaded successfully.")
else:
    print("Model file already exists.")



# Load the model
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image data from the request
    img_data = request.files['image'].read()
    
    # Process the image data
    img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (150, 150))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
    
    # Return prediction result
    return jsonify({'result': result})

@app.route('/<name>')
def home(name):
    return render_template('index.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
