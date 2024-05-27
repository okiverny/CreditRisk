import os, sys, glob

import pickle
import joblib
import datetime

import joblib
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

import gc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import lightgbm as lgb

#sys.path.append('/kaggle/input/predata-py/')
import predata


def compute_weight(data):
    df = data.select(['case_id','WEEK_NUM','target']).to_pandas()
    
    weights = pd.Series(dtype=float)  # Initialize an empty Series to store weights

    for week_num, group in df.groupby('WEEK_NUM'):
        weights_week = group['target'].value_counts(normalize=False)
        #weights_week = 1.0/weights_week
        weights_week = 1.0/weights_week*weights_week.to_dict()[1]
        weights = pd.concat([weights, group['target'].map(weights_week)])  # Concatenate weights for current group

    df['weights']=weights
    data = data.join(pl.from_pandas(df).drop(['WEEK_NUM','target']), on="case_id", how='left')
    return data

# Load data from parquet files
data_store = pl.read_parquet('../results_CreditRisk/data_store.parquet')
encoded_columns = joblib.load('../results_CreditRisk/encoded_columns.pkl')
cols_pred = [col for col in data_store.columns if col not in ['target','case_id','WEEK_NUM','MONTH','date_decision','weights']]

print('Number of encoded columns:',len(encoded_columns))
print('Number of all features (numerical and encoded):',len(cols_pred))

# Add class weights
data_store = compute_weight(data_store)

# Convert to pandas DataFrame
df_train = data_store.to_pandas()
del data_store

# Split into features and target
y = df_train["target"]
weeks = df_train["WEEK_NUM"]
weights = df_train["weights"]
df_train= df_train.drop(columns=['target','case_id','WEEK_NUM','MONTH','date_decision','weights'])


# Prepare LightGBM for training
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 16, 
    'num_leaves':62,
    #'min_data_in_leaf': 1932,
    "learning_rate": 0.016,
    "n_estimators": 3000,
    "colsample_bytree": 0.77,
    'subsample': 0.9,
    'subsample_freq': 2,
    "random_state": 42,
    "reg_alpha": 0.02,
    "reg_lambda": 0.5,
    #"extra_trees":True,
    "device": "gpu",
    'n_jobs': -1,
    "verbose": -1,
}

fitted_models = []
cv_scores = []
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks):#   Because it takes a long time to divide the data set, 
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
    # class weights
    weights_train, weights_valid = weights[idx_train], weights[idx_valid]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        categorical_feature=encoded_columns,
        eval_set = [(X_valid, y_valid)],
        eval_sample_weight = [weights_valid],
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
    fitted_models.append(model)
    y_pred_valid = model.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_valid, sample_weight=weights_valid)
    cv_scores.append(auc_score)
    
print("CV AUC scores: ", cv_scores)
print("AVG CV AUC score: ", np.mean(cv_scores))
print("Maximum CV AUC score: ", max(cv_scores))


joblib.dump(fitted_models, 'lgb_models.joblib')
joblib.dump(cols_pred, 'cols_pred.pkl')