from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import datetime

app = Flask(__name__)

# Load the trained model (use your model file path here)
model = load_model('mlp_water_quality.pkl')

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
    
    # Convert sensor data to numpy array and predict
    prediction_input = np.array(sensor_data).reshape(1, -1)
    prediction_percentage = model.predict(prediction_input)[0][0]  # Assuming a single output
    
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
