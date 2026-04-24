from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('../model/model.pkl', 'rb'))

@app.route('/')
def home():
    return "✅ Churn Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Expected order of features
        features = [
            data['Age'],
            data['Gender'],
            data['Tenure'],
            data['Usage Frequency'],
            data['Support Calls'],
            data['Payment Delay'],
            data['Subscription Type'],
            data['Contract Length'],
            data['Total Spend'],
            data['Last Interaction']
        ]

        prediction = model.predict([features])

        result = "Churn" if prediction[0] == 1 else "No Churn"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)