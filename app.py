from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = os.urandom(24)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load trained models
lgbm_model = joblib.load('lgbm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

MODEL_FEATURES = list(pd.read_csv('processed_X_train.csv').columns)
POTENTIAL_LABEL_COLUMNS = ['Class', 'Label', 'Fraud', 'IsFraud', 'Target']

def preprocess_input(data):
    for col in MODEL_FEATURES:
        if col not in data.columns:
            data[col] = 0
    return data[MODEL_FEATURES]

def convert_indices_to_ranges(indices):
    if not indices:
        return ''
    indices.sort()
    ranges = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = idx
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ', '.join(ranges)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', know_label=True)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    model_choice = request.form.get('model_choice')
    file = request.files['file']

    if not file:
        return render_template('index.html', error="No file uploaded")

    data = pd.read_csv(file)

    # Check for potential label columns
    label_column_found = None
    for potential_label in POTENTIAL_LABEL_COLUMNS:
        if potential_label in data.columns:
            label_column_found = potential_label
            break

    if label_column_found:
        y_true = data[label_column_found].values
        data.drop(columns=[label_column_found], inplace=True)
    else:
        y_true = None

    processed_data = preprocess_input(data)

    if model_choice == 'lgbm':
        model = lgbm_model
    elif model_choice == 'xgb':
        model = xgb_model
    elif model_choice == 'rf':
        model = rf_model
    else:
        return render_template('index.html', error="Invalid model choice")

    predictions = model.predict(processed_data)

    fraud_indices = [index for index, value in enumerate(predictions) if value == 1]

    # If y_true is available, compute metrics
    if y_true is not None:
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        
        # Confusion matrix visualization
        cm = confusion_matrix(y_true, predictions)
        z = cm[::-1]
        x = ['Not Fraud', 'Fraud']
        y = x[::-1]
        fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')
        cm_html = fig_cm.to_html(full_html=False)
    else:
        accuracy, precision, recall, cm_html = None, None, None, None

    # Probability histogram visualization for all models
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(processed_data)[:,1]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=y_prob, name='Predicted Probabilities'))
        fig_hist.update_layout(title_text='Distribution of Predicted Probabilities',
                               xaxis_title_text='Probability',
                               yaxis_title_text='Count')
        hist_html = fig_hist.to_html(full_html=False)
    else:
        hist_html = None

    return render_template('results.html', 
                           fraud_indices=fraud_indices, 
                           fraud_count=len(fraud_indices), 
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           cm_html=cm_html,
                           hist_html=hist_html)


                           

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def model_choice_to_model(choice):
    if choice == 'lgbm':
        return lgbm_model
    elif choice == 'xgb':
        return xgb_model
    elif choice == 'rf':
        return rf_model
    else:
        raise ValueError("Invalid model choice")

def compute_metrics(y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

def generate_confusion_matrix(y_true, predictions):
    cm = confusion_matrix(y_true, predictions)
    z = cm[::-1]
    x = ['Not Fraud', 'Fraud']
    y = x[::-1]
    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')
    return fig_cm.to_html(full_html=False)

def generate_histogram(predictions):
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=predictions, name='Predicted Classes'))
    fig_hist.update_layout(title_text='Distribution of Predicted Classes',
                           xaxis_title_text='Class',
                           yaxis_title_text='Count')
    return fig_hist.to_html(full_html=False)

if __name__ == "__main__":
    app.run(debug=True)
