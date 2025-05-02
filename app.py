from flask import Flask, request, jsonify
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load the trained models and their corresponding scalers
mlp_model = joblib.load('mlp_water_quality.pkl')
mlp_scaler = joblib.load('scaler_MLP.pkl')  # MLP Scaler

dnn_model = tf.keras.models.load_model('enhanced_water_quality_nn.h5')  # DNN model (TensorFlow)
dnn_scaler = joblib.load('scaler_dnn.pkl')  # DNN Scaler

knn_model = joblib.load('knn_water_quality.pkl')  # KNN model
knn_scaler = joblib.load('scaler_knn.pkl')  # KNN Scaler

log_reg_model = joblib.load('logistic_regression_model.pkl')  # Logistic Regression model
log_reg_scaler = joblib.load('scaler_logisticR.pkl')  # Logistic Regression Scaler

rf_model = joblib.load('random_forest_water_quality.pkl')  # Random Forest model
rf_scaler = joblib.load('scaler_RandomF.pkl')  # Random Forest Scaler

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

    # Scale the features using the respective scalers for each model
    mlp_features_scaled = mlp_scaler.transform([features])
    dnn_features_scaled = dnn_scaler.transform([features])
    knn_features_scaled = knn_scaler.transform([features])
    log_reg_features_scaled = log_reg_scaler.transform([features])
    rf_features_scaled = rf_scaler.transform([features])

    # Predict with each model
    mlp_proba = mlp_model.predict_proba(mlp_features_scaled)[0][1]  # MLP prediction (class 1)
    dnn_proba = dnn_model.predict(dnn_features_scaled)[0][0]  # DNN prediction (class 1)
    knn_proba = knn_model.predict_proba(knn_features_scaled)[0][1]  # KNN prediction (class 1)
    log_reg_proba = log_reg_model.predict_proba(log_reg_features_scaled)[0][1]  # Logistic Regression prediction (class 1)
    rf_proba = rf_model.predict_proba(rf_features_scaled)[0][1]  # Random Forest prediction (class 1)

    # Convert to percentage
    mlp_percentage = round(mlp_proba * 100, 2)
    dnn_percentage = round(dnn_proba * 100, 2)
    knn_percentage = round(knn_proba * 100, 2)
    log_reg_percentage = round(log_reg_proba * 100, 2)
    rf_percentage = round(rf_proba * 100, 2)

    # Store the last prediction with its parameters
    last_prediction = {
        "mlp_potability_percentage": mlp_percentage,
        "dnn_potability_percentage": dnn_percentage,
        "knn_potability_percentage": knn_percentage,
        "log_reg_potability_percentage": log_reg_percentage,
        "rf_potability_percentage": rf_percentage,
        "parameters": data
    }

    # Return all predictions in JSON format
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
