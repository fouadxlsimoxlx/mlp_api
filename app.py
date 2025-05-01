from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("mlp_water_quality.pkl")

@app.route('/predict', methods=['POST'])
def predict():
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
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([features])[0][1]
        percentage = round(proba * 100, 2)
        return jsonify({"potability_percentage": percentage})
    else:
        return jsonify({"error": "Model does not support predict_proba"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
