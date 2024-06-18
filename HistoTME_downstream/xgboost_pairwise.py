import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import matplotlib.pylab as plt
import shap
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
    df = df.drop(columns=ft_cols)
    return df

def main(seed):
    df = load_dataset(name='tme_clin')
    df_pdl1 = df[['ID', 'PDL1_scores']]
    df = df.iloc[:,:32]
    
    out_df = df.copy()

    if 'ID' in df.columns:
        df = df.drop(columns='ID')

    X = df.drop(['response_label'], axis=1)
    Y = df[['response_label']]
    y_encoded = OrdinalEncoder().fit_transform(Y)

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=seed, stratify=y_encoded, test_size=test_size)

    # Scale before making pairwise interactions
    scaler = StandardScaler().fit(X_train).set_output(transform='pandas')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = make_pairwise_cols(X_train_scaled)
    X_test_scaled = make_pairwise_cols(X_test_scaled)

    features = rf_selection(X_train_scaled, y_train.ravel(), n=18) 
    print(len(features), features[:10])

    dtrain_clf = xgb.DMatrix(X_train_scaled[features], y_train, enable_categorical=True)
    dtest_clf = xgb.DMatrix(X_test_scaled[features], y_test, enable_categorical=True)

    out_df = out_df.loc[X_test_scaled.index]

    n=1000
    params = {"objective": "multi:softprob", "tree_method": "hist", "num_class": 2, "device":"cuda:1",
                'eta':0.1, 'gamma':0.1}

    results = xgb.cv(
        params, dtrain_clf,
        num_boost_round=n,
        nfold=5,
        metrics=["mlogloss", 'auc', "merror"],
        seed=seed,
        early_stopping_rounds=100,
    )
    
    best_trees = np.argmax(results['test-auc-mean'])
    best_auc = results['test-auc-mean'].max()
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

    # Interpretability analysis with SHAP plots
    if not os.path.exists('interpretability'):
        os.mkdir('interpretability')

    gain_dict = model.get_score(importance_type='gain')
    feat_imp = pd.Series(gain_dict).sort_values(ascending=False)
    feat_imp.plot(kind='barh', title='Feature Importances', color='peru')
    plt.xlabel('Feature Importance Score')
    plt.gca().invert_yaxis()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    feat_imp.to_csv('interpretability/ft_impt_xgboost.csv')
    plt.savefig('interpretability/ft_impt_xgboost.png', bbox_inches="tight")
    plt.close()

    explainer = shap.TreeExplainer(model)
    X = pd.concat([X_train_scaled, X_test_scaled])
    shap_values = explainer.shap_values(X[features])
    shap.summary_plot(shap_values[1], X[features], show=False, plot_type='violin', max_display=10)
    f = plt.gcf()
    f.set_size_inches(10, 8)
    f.savefig('interpretability/shap_violinplot.png', bbox_inches='tight', dpi=500)
    plt.close()

    shap.summary_plot(shap_values[1], X[features], show=False, plot_type='dot', max_display=10)
    f = plt.gcf()
    f.set_size_inches(10, 8)
    f.savefig('interpretability/shap_dotplot.png', bbox_inches='tight', dpi=500)
    plt.close()

    row = 0
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][row], 
                                         base_values=explainer.expected_value[1], data=X[features].iloc[row],  
                                         feature_names=X[features].columns.tolist()))
    f = plt.gcf()
    f.set_size_inches(10, 7)
    f.savefig(f'interpretability/shap_waterfall_{row}.png', bbox_inches='tight', dpi=500)
    plt.close()

    row = 1
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][row], 
                                         base_values=explainer.expected_value[1], data=X[features].iloc[row],  
                                         feature_names=X[features].columns.tolist()))
    f = plt.gcf()
    f.set_size_inches(10, 7)
    f.savefig(f'interpretability/shap_waterfall_{row}.png', bbox_inches='tight', dpi=500)
    plt.close()


    shap.summary_plot(shap_values[1], X[features], show=False, plot_type="bar", max_display=10)
    g = plt.gcf()
    g.set_size_inches(16, 7)
    g.savefig('interpretability/shap_barplot.png', bbox_inches='tight', dpi=500)
    plt.close()

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

    out_df['prediction'] = 999
    for i in range(len(y_prob[:,1])):
        out_df['prediction'].iloc[i] = y_prob[:,1][i]

    out_df.to_csv('test_set_prediction.csv',index=False)

if __name__ == "__main__":
    main(seed=2)

