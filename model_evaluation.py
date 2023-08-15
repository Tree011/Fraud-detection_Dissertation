import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib

# Load the models
lgbm_model = joblib.load('lgbm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Load the test data
X_test = pd.read_csv('processed_X_test.csv')
y_test = pd.read_csv('processed_y_test.csv', header=None).iloc[:, 0]




def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', model_name="model"):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    labels = ['Non-Fraud', 'Fraud']
    plt.xticks(np.arange(2), labels)
    plt.yticks(np.arange(2), labels)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], 'd'), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_pred_prob, title='ROC Curve', model_name="model"):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob, title="Precision-Recall Curve", model_name="model"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{title}: AP={average_precision:0.2f}')
    plt.savefig(f'{model_name}_precision_recall_curve.png')
    plt.show()

# Evaluation function
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred, title=f'{model_name} Confusion Matrix', model_name=model_name)
    plot_roc_curve(y_test, y_pred_prob, title=f'{model_name} ROC Curve', model_name=model_name)
    plot_precision_recall_curve(y_test, y_pred_prob, title=f"{model_name} Precision-Recall Curve", model_name=model_name)

# Evaluate all models
evaluate_model(lgbm_model, "LGBM")
evaluate_model(xgb_model, "XGBoost")
evaluate_model(rf_model, "RandomForest")
