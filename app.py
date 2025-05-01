from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Example: Loading the model (adjust according to your model)
model = joblib.load('path_to_your_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the JSON data sent by the client
        # Extract parameters and ensure they are correct
        features = [
            data["pH"],
            data["Turbidity"],
            data["Hardness"],
            data["Solids"],
            data["Chloramines"],
            data["Sulfate"],
            data["Conductivity"],
            data["TOC"],
            data["THMs"]
        ]
        prediction = model.predict([features])  # Predict using the model
        result = prediction[0]
        return jsonify({"prediction": result}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred, please check the server logs."}), 500

if __name__ == '__main__':
    app.run(debug=True)
