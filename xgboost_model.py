import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
X_train = pd.read_csv('processed_X_train.csv')
y_train = pd.read_csv('processed_y_train.csv', header=None).iloc[:, 0]
X_test = pd.read_csv('processed_X_test.csv')
y_test = pd.read_csv('processed_y_test.csv', header=None).iloc[:, 0]

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_smote, y_smote, eval_metric='logloss')

# Save the model
joblib.dump(xgb, 'xgboost_model.pkl')

# Evaluate on test set
y_pred_test = xgb.predict(X_test)
print("XGBoost Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test))
