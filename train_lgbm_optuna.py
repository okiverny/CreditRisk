import os, sys, glob

import pickle
import joblib
import datetime

import joblib
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

import optuna

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

#data_store = data_store.sample(n=100000, seed=42)

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
#splits = cv.split(df_train, y, groups=weeks)

def lgbm_objective(trial):
    """
    LGBMClassifier parameters search
    """
    # Target ratio for unbalanced data
    #y_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    
    params = {
        'n_estimators': 2500,
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        #'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        #'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-4, 10),
        #'reg_lambda': trial.suggest_loguniform('lambda_l2', 0.1, 10),
        'colsample_bytree': trial.suggest_float('feature_fraction', 0.5, 1),
        'subsample': trial.suggest_float('bagging_fraction', 0.5, 1),
        'subsample_freq': trial.suggest_int('bagging_freq', 0, 10),
        
        'objective': 'binary',
        'metric': 'auc',
        #'scale_pos_weight': y_ratio,
        'verbosity': -1,
        #'device': 'gpu',
        'learning_rate': 0.04,
        "reg_alpha": 0.02,
        "reg_lambda": 0.5,
        'max_bin': 255,
        'n_jobs': -1,
    }

    cv_scores = []
    for idx_train, idx_valid in cv.split(df_train, y, groups=weeks):
        X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
        weights_train, weights_valid = weights[idx_train], weights[idx_valid]
    
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
              sample_weight=weights_train,
              eval_set = [(X_valid, y_valid)],
              eval_sample_weight = [weights_valid],
              categorical_feature=encoded_columns,
              #eval_metric=stability_metric,
              callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
              )
        
        y_pred = model.predict_proba(X_valid)[:, 1]
        #_, stability_score, _ = stability_metric(np.array(y_valid), np.array(y_pred))
        score = roc_auc_score(np.array(y_valid), np.array(y_pred))
        cv_scores.append(score)

    return np.mean(cv_scores)


# Optuna study
objective = lgbm_objective
sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)

# Show best results
trial = study.best_trial

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
print('Value:', trial.value)
print('Params:')

for key, value in trial.params.items():
    print('{}: {}'.format(key, value))
