from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore



# Initialize Firebase Admin SDK
cred = credentials.Certificate('google-services.json')  # Replace with your actual filename
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()


app = Flask(__name__)

# Load the trained models and their corresponding scalers
mlp_model = joblib.load('mlp_water_quality.pkl')
mlp_scaler = joblib.load('scaler_MLP.pkl')  # MLP Scaler

#dnn_model = load_model('dnn_water_quality_model.h5')
dnn_model = tf.keras.models.load_model('enhanced_water_quality_nn.h5')  # DNN model (commented out)
dnn_scaler = joblib.load('scaler_dnn.pkl')  # DNN Scaler (commented out)

knn_model = joblib.load('knn_water_quality.pkl')
knn_scaler = joblib.load('scaler_knn.pkl')

log_reg_model = joblib.load('logistic_regression_model.pkl')
log_reg_scaler = joblib.load('scaler_logisticR.pkl')

rf_model = joblib.load('random_forest_water_quality.pkl')
rf_scaler = joblib.load('scaler_RandomF.pkl')

# In-memory storage for the last prediction
last_prediction = {}

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction

    data = request.get_json()

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

    # Scale features using respective scalers
    mlp_features_scaled = mlp_scaler.transform([features])
    dnn_features_scaled = dnn_scaler.transform([features])  # Commented out
    knn_features_scaled = knn_scaler.transform([features])
    log_reg_features_scaled = log_reg_scaler.transform([features])
    rf_features_scaled = rf_scaler.transform([features])

    # Get predictions from each model
    mlp_proba = mlp_model.predict_proba(mlp_features_scaled)[0][1]
    dnn_proba = dnn_model.predict(dnn_features_scaled)[0][0]  # Commented out
    knn_proba = knn_model.predict_proba(knn_features_scaled)[0][1]
    log_reg_proba = log_reg_model.predict_proba(log_reg_features_scaled)[0][1]
    rf_proba = rf_model.predict_proba(rf_features_scaled)[0][1]

    # Convert probabilities to percentages
    mlp_percentage = round(mlp_proba * 100, 2)
    dnn_percentage = round(dnn_proba * 100, 2)  # Commented out
    knn_percentage = round(knn_proba * 100, 2)
    log_reg_percentage = round(log_reg_proba * 100, 2)
    rf_percentage = round(rf_proba * 100, 2)

    last_prediction = {
        "mlp_potability_percentage": mlp_percentage,
        "dnn_potability_percentage": dnn_percentage,  # Commented out
        "knn_potability_percentage": knn_percentage,
        "log_reg_potability_percentage": log_reg_percentage,
        "rf_potability_percentage": rf_percentage,
        "parameters": data
    }

    # Store the result in Firestore
    db.collection('predictions').add({
        'ph': data['ph'],
        'Hardness': data['Hardness'],
        'Solids': data['Solids'],
        'Chloramines': data['Chloramines'],
        'Sulfate': data['Sulfate'],
        'Conductivity': data['Conductivity'],
        'Organic_carbon': data['Organic_carbon'],
        'Trihalomethanes': data['Trihalomethanes'],
        'Turbidity': data['Turbidity'],
        'mlp_potability_percentage': mlp_percentage,
        'dnn_potability_percentage': dnn_percentage,
        'knn_potability_percentage': knn_percentage,
        'log_reg_potability_percentage': log_reg_percentage,
        'rf_potability_percentage': rf_percentage,
        'timestamp': firestore.SERVER_TIMESTAMP,
    })
    
    return jsonify(last_prediction)

@app.route('/last_prediction', methods=['GET'])
def get_last_prediction():
    if last_prediction:
        return jsonify(last_prediction)
    else:
        return jsonify({"message": "No prediction has been made yet."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
