from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('mlp_water_quality.pkl')
scaler = joblib.load('scaler_MLP.pkl')  # Load the scaler as well

# In-memory storage for the last prediction
last_prediction = {}

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction

    # Get JSON data from the request
    data = request.get_json()

    # Extract features from the request
    features = [
        data['ph'],
        data['Hardness'],
        data['Solids'],
        data['Chloramines'],
        data['Sulfate'],
        data['Conductivity'],
        data['Organic_carbon'],
        data['Trihalomethanes'],
        data['Turbidity']
    ]

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform([features])

    # Get probability for class 1 (potable) from the model
    proba = model.predict_proba(features_scaled)[0][1]  # [0][1] → first sample, class 1

    # Convert to percentage
    percentage = round(proba * 100, 2)

    # Store the last prediction with its parameters
    last_prediction = {
        "potability_percentage": percentage,
        "parameters": data
    }

    # Return the percentage in JSON format
    return jsonify(last_prediction)

# Define the /last_prediction endpoint
@app.route('/last_prediction', methods=['GET'])
def get_last_prediction():
    if last_prediction:
        return jsonify(last_prediction)
    else:
        return jsonify({"message": "No prediction has been made yet."}), 404

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
