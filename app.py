from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON data
        data = request.get_json()

        # Perform prediction (you can call your model here)
        # For the sake of this example, let's just return a mock value
        prediction = 75.0  # Replace this with actual prediction logic

        # Send back the result
        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong!"}), 500

if __name__ == '__main__':
    app.run(debug=True)
