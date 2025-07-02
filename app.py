
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        math = float(request.form['math'])
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])

        features = np.array([[math, reading, writing]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Prediction: {prediction}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
    
