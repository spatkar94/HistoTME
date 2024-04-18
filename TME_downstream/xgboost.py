import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import shap
from tqdm import tqdm
import itertools
from utils import rf_selection
from data import load_dataset
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def main(seed):
    df = load_dataset(name='tme_clin')
    df = df.iloc[:,:32]

    if 'ID' in df.columns:
        df = df.drop(columns='ID')

    print(df)

    X = df.drop(columns='response_label')
    Y = df[['response_label']]
    y_encoded = OrdinalEncoder().fit_transform(Y)

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=seed, stratify=y_encoded, test_size=test_size)

    features = rf_selection(X_train, y_train.ravel(), n=10)
    print(len(features), features)
    X_train = X_train[features]
    X_test = X_test[features]

    dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    n=1000
    params = {"objective": "multi:softprob", "tree_method": "hist", "num_class": 2, 'eta':0.1, 'max_depth':3}
    params["device"] = "cuda:3"
    results = xgb.cv(
        params, dtrain_clf,
        num_boost_round=n,
        nfold=10,
        metrics=["mlogloss", 'auc', "merror"],
        seed=seed,
        stratified=True,
        early_stopping_rounds=100,
    )
    
    best_trees = np.argmax(results['test-auc-mean'])
    print(f'best trees occurs at {best_trees}')
    print('Train AUC = ', results['train-auc-mean'][best_trees])
    print('Train error = ', results['train-merror-mean'][best_trees])
    print('CV AUC = ', results['test-auc-mean'].max())
    print('CV error = ', results['test-merror-mean'].max())

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
