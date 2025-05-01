from flask import Flask, request, jsonify
import numpy as np
import joblib  # Use joblib for loading the model and scaler
import datetime

app = Flask(__name__)

# Load the trained model and scaler (use your model and scaler file paths here)
model = joblib.load('mlp_water_quality.pkl')  # Load model with joblib
scaler = joblib.load('scaler.pkl')  # Load scaler with joblib (if you saved it)

# Store data temporarily in a list or dictionary
predictions_data = []

# Function to predict water quality
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract sensor data (parameters)
    sensor_data = [
        data['pH'], data['Turbidity'], data['Hardness'], data['Chloramines'], data['Sulfate'],
        data['Conductivity'], data['Organic_carbon'], data['Trihalomethanes']
    ]
    
    # Convert sensor data to numpy array
    sensor_data_array = np.array(sensor_data).reshape(1, -1)
    
    # Scale the sensor data using the scaler
    scaled_data = scaler.transform(sensor_data_array)
    
    # Predict the water quality
    prediction_percentage = model.predict(scaled_data)[0]  # Assuming a single output
    
    # Store data temporarily
    predictions_data.append({
        'parameters': data,
        'prediction_percentage': prediction_percentage,
        'timestamp': str(datetime.datetime.now())
    })
    
    # Return prediction and parameters
    return jsonify({
        'prediction_percentage': prediction_percentage,
        'parameters': data
    })

# Endpoint to get the last prediction data (optional)
@app.route('/last_prediction', methods=['GET'])
def get_last_prediction():
    if predictions_data:
        return jsonify(predictions_data[-1])
    return jsonify({"message": "No prediction data available"}), 404

if __name__ == '__main__':
    app.run(debug=True)
