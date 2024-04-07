from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import os

app = Flask(__name__)

model_path = 'model/model.tflite'

# Load the model
try:
    model = Interpreter(model_path)
    model.allocate_tensors()
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    # Log the error and handle it gracefully
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # return jsonify({'result': "Predicting..."})

        # Receive image data from the request
        img_data = request.files['image'].read()
        
        # Process the image data
        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        img_array = cv2.resize(img_array, (150, 150))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Check if FLOAT32 or 64:
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        # Set the tensor (model input)
        input_details = model.get_input_details()
        model.set_tensor(input_details[0]['index'], img_array)
        
        # Run the model
        model.invoke()
        
        # Get the tensor (model output)
        output_details = model.get_output_details()
        prediction = model.get_tensor(output_details[0]['index'])
        
        result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
        
        # Return prediction result
        return jsonify({'result': result})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/')
def index():
    return render_template('index.html', name='World')

@app.route('/<name>')
def home(name):
    return render_template('index.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
