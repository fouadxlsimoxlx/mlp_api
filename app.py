from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("mlp_water_quality.pkl")

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
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

    # Return the prediction as JSON
    return jsonify({"potability": int(prediction)})

# Start the Flask app (make sure it runs on any available IP)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
