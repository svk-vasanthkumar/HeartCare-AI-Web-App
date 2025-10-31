from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('disease_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get('age')),
            float(request.form.get('sex')),
            float(request.form.get('cp')),
            float(request.form.get('trestbps')),
            float(request.form.get('chol')),
            float(request.form.get('fbs')),
            float(request.form.get('restecg')),
            float(request.form.get('thalach')),
            float(request.form.get('exang')),
            float(request.form.get('oldpeak')),
            float(request.form.get('slope')),
            float(request.form.get('ca')),
            float(request.form.get('thal'))
        ]

        prediction = model.predict([features])[0]
        result_text = "⚠️ Disease Detected! Please consult a doctor." if prediction == 1 else "😢 Heart Disease Detected!  Please consult a doctor"

        # Send data to result page
        details = {
            "Age": features[0],
            "Sex": "Male" if features[1] == 1 else "Female",
            "Chest Pain Type": features[2],
            "Resting BP": features[3],
            "Cholesterol": features[4],
            "Fasting Blood Sugar": features[5],
            "Rest ECG": features[6],
            "Max Heart Rate": features[7],
            "Exercise Angina": features[8],
            "Oldpeak": features[9],
            "Slope": features[10],
            "CA": features[11],
            "Thal": features[12],
        }

        return render_template("result.html", result=result_text, details=details)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
