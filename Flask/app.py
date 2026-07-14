import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and the fitted scaler
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'rdf.pkl'), 'rb'))
scale = pickle.load(open(os.path.join(os.path.dirname(__file__), 'scale1.pkl'), 'rb'))

# Must match the exact column order used during training
FEATURE_NAMES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["GET"])
def predict():
    return render_template('input.html')


@app.route('/submit', methods=["POST", "GET"])
def submit():
    # Read the inputs submitted by the user, in the same order as FEATURE_NAMES
    input_feature = [float(request.form[name]) for name in FEATURE_NAMES]

    data = pd.DataFrame([input_feature], columns=FEATURE_NAMES)

    # Apply the same scaling used during training
    scaled_data = scale.transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=FEATURE_NAMES)

    prediction = model.predict(scaled_data)[0]

    return render_template('output.html', prediction=int(prediction))


if __name__ == '__main__':
    app.run(debug=True)
