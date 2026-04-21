from flask import Flask, request, jsonify
import numpy as np
import os
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# FULL class names
class_names = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry___healthy',
'Cherry___Powdery_mildew',
'Corn___Cercospora_leaf_spot Gray_leaf_spot',
'Corn___Common_rust',
'Corn___healthy',
'Corn___Northern_Leaf_Blight',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___healthy',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___healthy',
'Potato___Late_blight',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___healthy',
'Strawberry___Leaf_scorch',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___healthy',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_mosaic_virus',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

@app.route('/')
def home():
    return "AgroAI API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Load image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        # Convert to array
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Run model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        label = class_names[class_index]

        if "___" in label:
            plant, disease = label.split("___")
        else:
            plant = label
            disease = "Unknown"

        return jsonify({
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
