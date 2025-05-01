from flask import Flask, request, jsonify
import joblib  # or keras.models.load_model if you're using Keras
import numpy as np

app = Flask(__name__)

# Temporary storage
last_result = {}

# Load your trained model
# model = joblib.load('mlp_water_quality.pkl')  # For sklearn models
# OR for Keras: model = load_model('enhanced_water_quality_nn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    global last_result

    data = request.get_json()

    try:
        # Extract 9 parameters
        features = [
            data['pH'],
            data['Hardness'],
            data['Solids'],
            data['Chloramines'],
            data['Sulfate'],
            data['Conductivity'],
            data['Organic_carbon'],
            data['Trihalomethanes'],
            data['Turbidity']
        ]

        # Model prediction (fake logic for now — replace with your real model)
        # prediction = model.predict([features])[0]
        # For demo, let's just simulate a value:
        prediction_percentage = 86.3  # Replace with your actual result

        # Store it
        last_result = {"potability": prediction_percentage}

        return jsonify(last_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_result', methods=['GET'])
def get_result():
    if not last_result:
        return jsonify({"message": "No result available yet."}), 404
    return jsonify(last_result), 200

if __name__ == '__main__':
    app.run()
