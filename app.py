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

    # Get probability for class 1 (potable) from the model
    proba = model.predict_proba([features])[0][1]  # [0][1] → first sample, class 1

    # Convert to percentage
    percentage = round(proba * 100, 2)

    # Return the percentage in JSON format
    return jsonify({"potability_percentage": percentage})

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
