import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from model import preprocess_text

app = Flask(__name__)


with open('maps_updated.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = joblib.load(encoder_file)

expected_features = ['Customer', 'Source', 'SourceType', 'FailType', 'FailSummary']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data['features']])

    if not all(feature in df.columns for feature in expected_features):
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        return jsonify({'error': 'Eksik özellikler var', 'missing_features': missing_features}), 400

    try:
        
        df['FailSummary'] = df['FailSummary'].apply(preprocess_text)
        df['FailType'] = df['FailType'].apply(preprocess_text)
        df['Customer'] = df['Customer'].apply(preprocess_text)
        df['Source'] = df['Source'].apply(preprocess_text)
        df['SourceType'] = df['SourceType'].apply(preprocess_text)

     
        predicted_label = model.predict(df)[0]
        predicted_solution = label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({'Solution': predicted_solution}), 200

    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({'error': 'Model tahmini sırasında bir hata oluştu'}), 500


if __name__ == '__main__':
    app.run(debug=True)
