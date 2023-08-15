from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained models
lgbm_model = joblib.load('lgbm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

MODEL_FEATURES = list(pd.read_csv('processed_X_train.csv').columns)

def preprocess_input(data):
    # Fill missing columns with zeros
    for col in MODEL_FEATURES:
        if col not in data.columns:
            data[col] = 0
    # Reorder columns to match model's expected order
    data = data[MODEL_FEATURES]
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model_choice')
    file = request.files['file']

    if not file:
        return "No file uploaded", 400

    data = pd.read_csv(file)
    processed_data = preprocess_input(data)

    if model_choice == 'lgbm':
        predictions = lgbm_model.predict(processed_data)
    elif model_choice == 'xgb':
        predictions = xgb_model.predict(processed_data)
    elif model_choice == 'rf':
        predictions = rf_model.predict(processed_data)
    else:
        return "Invalid model choice", 400

    fraud_indices = [index for index, value in enumerate(predictions) if value == 1]
    
    # For visualization purposes, we'll just return the fraud indices. 
    # In a real-world scenario, you might return the entire rows of fraudulent transactions.
    return jsonify({
        'fraud_indices': fraud_indices,
        'fraud_count': len(fraud_indices)
    })

if __name__ == "__main__":
    app.run(debug=True)
