import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer, classification_report
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
from helper import data_input_preprocess

import warnings
warnings.filterwarnings('ignore')

dt_inp = [{
    'gender': 'Male',
    'senior_citizen': 'Yes',
    'partner': 'Yes',
    'dependents': 'Yes',
    'tenure_months': 10,
    'phone_service': 'Yes',
    'multiple_lines': 'Yes',
    'internet_service': 'Fiber optic',
    'online_security': 'Yes',
    'online_backup': 'Yes',
    'device_protection': 'Yes',
    'tech_support': 'Yes',
    'streaming_tv': 'Yes',
    'streaming_movies': 'Yes',
    'contract': 'Month-to-month',
    'paperless_billing': 'Yes',
    'payment_method': 'Credit card (automatic)',
    'monthly_charges': 20,
    'total_charges': 200
}]

dt_inp = pd.DataFrame(dt_inp)

dt_inp = data_input_preprocess(dt_inp)
dt_inp = dt_inp.head(1).drop(columns=['churn_value'], axis=1)

gs_lr = joblib.load('./classifier.pkl')
y_pred_input = gs_lr.predict(dt_inp)
print(y_pred_input)