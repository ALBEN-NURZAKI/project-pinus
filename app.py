from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

app = Flask(__name__)

# Load dataset (contoh data dari Excel)
data = {
    "Circumference (m)": [0.3, 0.18, 0.46, 0.63, 0.23, 0.56, 0.39, 0.41, 0.62, 0.43, 0.15, 0.19, 0.17, 0.17, 0.22, 0.45, 0.39, 0.42, 0.38, 0.3, 0.18],
    "Height (m)": [7.21, 5.12, 8.83, 12.08, 5.81, 13.5, 10.9, 6.79, 10.66, 10.5, 2.67, 20.34, 19.72, 19.8, 23.7, 32.51, 26.23, 32.51, 29.18, 26.1, 21.51],
    "Species": ["Douglas Fir"]*11 + ["White Pine"]*10
}
df = pd.DataFrame(data)

# Train KNN Model
X = df[["Circumference (m)", "Height (m)"]]
y = df["Species"]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Certainty Factor Calculation
def calculate_cf(prediction, confidence):
    return min(confidence * 0.9, 0.95)  # Contoh sederhana

# Placeholder Computer Vision
def detect_pinus(image_path):
    return "Pinus Detected", 0.8  # Dummy response

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    circumference = float(data['circumference'])
    height = float(data['height'])
    
    # Prediksi KNN
    prediction = knn.predict([[circumference, height]])[0]
    proba = knn.predict_proba([[circumference, height]]).max()
    
    # Certainty Factor
    cf = calculate_cf(prediction, proba)
    
    response = {
        "species": prediction,
        "confidence": f"{cf:.2%}",
        "message": f"Jenis Pinus: {prediction} (Keyakinan: {cf:.2%})"
    }
    return jsonify(response)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"})
    
    # Simpan file sementara
    image_path = os.path.join('static', 'uploads', file.filename)
    file.save(image_path)
    
    # Placeholder CV
    result, confidence = detect_pinus(image_path)
    return jsonify({
        "result": result,
        "confidence": confidence,
        "message": f"{result} (Confidence: {confidence:.2%})"
    })

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)