from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
from random import randint
import pandas as pd
import random
import math

def rf_selection(X, y, n=None, weight=None):
    rfc = RandomForestClassifier(n_estimators = 100, random_state=1)
    if not n:
        sel = SelectFromModel(rfc)
    else:
        sel = SelectFromModel(rfc, max_features=n)
    sel.fit(X, y, sample_weight=weight)
    selected_feat= X.columns[(sel.get_support())]
    return selected_feat

def sfs_selection(X, y, n=None):
    # Sequential Forward Selection(sfs)
    if not n:
        n = 10
    sfs = SequentialFeatureSelector(LogisticRegression(),
            n_features_to_select=n,
            direction='forward',
            scoring = 'f1',
            cv = 5)
    sfs.fit(X, y)
    features = sfs.get_feature_names_out()
    return features

def lasso_selection(X,y):
    # Lasso selection
    #grid search for best C
    params = {"C":np.linspace(0.00001, 1, 500)}
    kf=KFold(n_splits=5,shuffle=True, random_state=42)

    lasso = LogisticRegression('l1', solver='liblinear')
    lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X, y)
    print("Best Params {}".format(lasso_cv.best_params_))
    C = lasso_cv.best_params_['C']

    lasso1 = LogisticRegression('l1',solver='liblinear', C=C)
    lasso1.fit(X, y)
    lasso1_coef = np.abs(lasso1.coef_).squeeze()
    features = []
    for i in range(len(X.columns)):
        if lasso1_coef[i] != 0:
            features.append(X.columns[i])
    return features

def get_groups():
    features = {}
    features['protumor'] = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg',
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic',
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
    features['antitumor'] = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells',
                   'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature',
                   'Antitumor_cytokines', 'IFNG'] # 12 features
    features['angio'] = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features
    features['cancer'] = ['Proliferation_rate', 'EMT_signature']
    return features

def choose_single_vs_multi(df_single, df_multi):
    # Multi performs better on (threshold -log(p) > 200):
    multitask_ft = ['MHCI', 'Coactivation_molecules', 'Effector_cells', 'T_cell_traffic', 'NK_cells',
                    'T_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 'Checkpoint_inhibition', 'T_reg_traffic',
                    'MDSC_traffic', 'Macrophages', 'Th2_signature', 'Protumor_cytokines', 'Endothelium', 'Proliferation_rate',
                    'EMT_signature', 'IFNG']
    df_single = df_single.drop(columns=multitask_ft)
    df_multi = df_multi[['ID'] + multitask_ft]

    return df_single.merge(df_multi, how='left', on='ID')

