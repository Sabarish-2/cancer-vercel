from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/tumor.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image = request.files['image']
    
    # Preprocess the uploaded image
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
