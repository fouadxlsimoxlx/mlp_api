from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model (make sure the model path is correct)
model = joblib.load("mlp_water_quality.pkl")

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Ensure all required parameters are provided
        required_params = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]

        # Check if all parameters are present in the request
        if not all(param in data for param in required_params):
            return jsonify({"error": "Missing parameters. Please provide all required parameters."}), 400

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

        # Check if model supports predict_proba (classification) or predict (regression)
        # Assuming it's a classification model (e.g., RandomForest, LogisticRegression)
        if hasattr(model, 'predict_proba'):
            # Get probability for class 1 (potable)
            proba = model.predict_proba([features])[0][1]  # [0][1] → first sample, class 1
            percentage = round(proba * 100, 2)
            return jsonify({"potability_percentage": percentage})
        
        # If it's a regression model, use predict to get continuous output
        elif hasattr(model, 'predict'):
            prediction = model.predict([features])[0]  # Regression model prediction
            # Optionally, you could scale this into a percentage if needed
            return jsonify({"potability_percentage": round(prediction, 2)})
        
        else:
            return jsonify({"error": "Model does not support prediction."}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
