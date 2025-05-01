from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("mlp_water_quality.pkl")

# Store the prediction in memory (or you can use a database for persistence)
stored_prediction = None

# Define the /predict endpoint (POST) to send parameters and calculate the result
@app.route('/predict', methods=['POST'])
def predict():
    global stored_prediction  # Make sure we're modifying the global variable
    
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
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Calculate percentage (for example, multiply by 100 if you need it as percentage)
    prediction_percentage = prediction * 100
    
    # Store the result for later retrieval (if needed)
    stored_prediction = prediction_percentage
    
    # Return the prediction as JSON
    return jsonify({"prediction_percentage": prediction_percentage}), 200


# Define the /get_prediction endpoint (GET) to retrieve the prediction
@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    if stored_prediction is None:
        return jsonify({"error": "No prediction available. Please send parameters using POST first."}), 400
    return jsonify({"prediction_percentage": stored_prediction}), 200


# Start the Flask app (make sure it runs on any available IP)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
