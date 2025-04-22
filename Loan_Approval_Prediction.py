from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import sys

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\rapol\vs code\pending\LOAN APPROVAL PREDICTION\LOAN APPROVAL PREDICTION\test_case\ab_best_model.pkl'

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(2)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Credit_History = float(request.form['Credit_History'])
        Property_Area = float(request.form['Property_Area'])
        Income = float(request.form['Income'])

        # Prepare data for prediction
        data = np.array([[Credit_History, Property_Area, Income]])

        # Make prediction
        prediction = model.predict(data)[0]
        result = "Loan Approved" if prediction == 1 else "Loan Rejected"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    try:
        app.run(debug=False)
    except SystemExit as e:
        print(f"Exited with SystemExit: {e.code}")
        sys.exit(e.code)
