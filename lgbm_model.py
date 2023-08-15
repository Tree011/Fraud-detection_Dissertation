import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
X_train = pd.read_csv('processed_X_train.csv')
y_train_df = pd.read_csv('processed_y_train.csv', header=None)
y_train = y_train_df.iloc[:, 0]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Train LGBM model
lgbm = LGBMClassifier()
lgbm.fit(X_smote, y_smote)

# Save the model
joblib.dump(lgbm, 'lgbm_model.pkl')

# Load test data to evaluate
X_test = pd.read_csv('processed_X_test.csv')
y_test_df = pd.read_csv('processed_y_test.csv', header=None)
y_test = y_test_df.iloc[:, 0]
y_pred = lgbm.predict(X_test)

# Evaluation
print("LGBM Classification Report (Test Set):")
print(classification_report(y_test, y_pred))
