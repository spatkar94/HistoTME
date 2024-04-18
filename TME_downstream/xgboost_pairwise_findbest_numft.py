import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import matplotlib.pylab as plt
import json
from tqdm import tqdm
import itertools
import os
from utils import rf_selection
from data import load_dataset
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def make_pairwise_cols(df):
    ft_cols = df.columns
    column_pairs = list(itertools.combinations(ft_cols, 2))
    for col1, col2 in column_pairs:
        df[f'{col1}__{col2}__sum'] = df[col1] + df[col2]
        df[f'{col1}__{col2}__mul'] = np.exp(df[col1]) * np.exp(df[col2])
        df[f'{col1}__{col2}__sub'] = df[col1] - df[col2]
        df[f'{col1}__{col2}__div'] = np.exp(df[col1]) / np.exp(df[col2])
    return df

def main(seed):
    df = load_dataset(name='tme_clin')
    df = df.iloc[:,:32]
 
    if 'ID' in df.columns:
        df = df.drop(columns='ID')

    print(df)

    X = df.drop(['response_label'], axis=1)
    Y = df[['response_label']]
    y_encoded = OrdinalEncoder().fit_transform(Y)

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=seed, stratify=y_encoded, test_size=test_size)

    scaler = StandardScaler().fit(X_train).set_output(transform='pandas')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = make_pairwise_cols(X_train_scaled)
    X_test_scaled = make_pairwise_cols(X_test_scaled)

    max_ft = len(rf_selection(X_train_scaled, y_train.ravel()))
    num_ft_choices = np.arange(1,100, 1)
    best_num_ft = 0
    best_trees = 0
    best_auc = 0
    auc_dict = {}
    for num_ft in num_ft_choices:
        features = rf_selection(X_train_scaled, y_train.ravel(), n=num_ft)
        print(len(features), features[:10])

        dtrain_clf = xgb.DMatrix(X_train_scaled[features], y_train, enable_categorical=True)
        dtest_clf = xgb.DMatrix(X_test_scaled[features], y_test, enable_categorical=True)

        n=1000
        params = {"objective": "multi:softprob", "tree_method": "hist", "num_class": 2, 'eta':0.1, 'max_depth':3}
        params["device"] = "cuda:1"
        results = xgb.cv(
            params, dtrain_clf,
            num_boost_round=n,
            nfold=10,
            metrics=["mlogloss", 'auc', "merror"],
            seed=seed,
            stratified=True,
            early_stopping_rounds=100,
        )
       
        best_trees_new = np.argmax(results['test-auc-mean'])
        print(f'best trees occurs at {best_trees_new}')
        print('CV AUC = ', results['test-auc-mean'].max())
        print('CV error = ', results['test-merror-mean'].max())

        if results['test-auc-mean'].max() > best_auc:
            best_auc = results['test-auc-mean'].max()
            best_num_ft = num_ft
            best_trees = best_trees_new

        auc_dict[num_ft] = results['test-auc-mean'].max()

    print('') 
    print(f'Best num_ft = {best_num_ft}')
    print(f'Best num_trees = {best_trees}')
    print(f'CV AUC = {best_auc}')

    df_auc = pd.DataFrame.from_dict(auc_dict, orient='index', columns=['ROC_AUC'])
    df_auc = df_auc.reset_index().rename(columns={'index':'num_ft'})
    
    if not os.path.exists('cv_auc'):
        os.mkdir('cv_auc')
    df_auc.to_csv('cv_auc/pairwise_ft_selection_cv_auc.csv', index=False)

    features = rf_selection(X_train_scaled, y_train.ravel(), n=best_num_ft)
    print('')
    print(len(features), features[:10])
    
    dtrain_clf = xgb.DMatrix(X_train_scaled[features], y_train, enable_categorical=True)
    dtest_clf = xgb.DMatrix(X_test_scaled[features], y_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain_clf,
        num_boost_round=best_trees,
    )

    y_prob = model.predict(dtest_clf)
    y_pred = np.argmax(y_prob, axis=1)

    scores = [accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob[:,1]), precision_score(y_test, y_pred), 
                recall_score(y_test, y_pred), f1_score(y_test, y_pred), average_precision_score(y_test, y_pred)]

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, y_prob[:,1]))
    print("Test Precision:", precision_score(y_test, y_pred))
    print("Test Recall:", recall_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))
    print("Test AP:", average_precision_score(y_test, y_pred))

if __name__ == "__main__":
    main(seed=42)


