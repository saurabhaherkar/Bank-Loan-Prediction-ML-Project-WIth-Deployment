import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)
model = joblib.load('Bank Loan Prediction.pkl')


@app.route('/')
def home():
    return render_template('index_bank.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_features = [int(x) for x in request.form.values()]
        feature_values = [np.array(input_features)]
        feature_names = ['age', 'experience', 'income', 'family', 'education']

        df = pd.DataFrame(feature_values, columns=feature_names)
        print(df)
        out = model.predict(df)
        print(out)
        if out == 0:
            res = 'Congratulations, Loan Approved!!'
        else:
            res = 'Sorry, Loan Denied'

        return render_template('index_bank.html', prediction_text=res)
    else:
        return render_template('index_bank.html')


if __name__ == '__main__':
    app.run(debug=True)
