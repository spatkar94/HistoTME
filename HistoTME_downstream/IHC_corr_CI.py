import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from utils import choose_single_vs_multi

cor_type = 'spearman'
if cor_type == 'pearson':
    cor = pearsonr
elif cor_type == 'spearman':
    cor = spearmanr 

def remove_parentheses(input_string):
    result = re.sub(r'\([^)]*\)', '', input_string)
    return result

def clean_ihc_data(df, name):
    df['Image'] = df['Image'].apply(lambda x: remove_parentheses(x))
    df = df[['Image', 'Num Positive per mm^2']]
    df['ID'] = df['Image'].apply(lambda x: x[0:14])
    df_mean = df.groupby(['ID']).mean(numeric_only=True).reset_index().rename(columns={'Num Positive per mm^2':f'{name}_count'})
    df = df.merge(df_mean, how='left', on='ID')
    df = df[['ID', f'{name}_count']].drop_duplicates()
    return df

def bootstrap_ihc():
    df_single = pd.read_csv('../TME_regression/predictions/suny_predictions_singletask_UNI.csv')
    df_multi = pd.read_csv('../TME_regression/predictions/suny_predictions_multitask_UNI.csv')
    df = choose_single_vs_multi(df_single, df_multi)

    df_cd4 = pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AlexChen/suny_project/ihc_measurements/SUNY_cd8_tcell_qupath_measurements.csv')
    df_cd8 = pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AlexChen/suny_project/ihc_measurements/SUNY_cd4_tcell_qupath_measurements.csv')
    df_bcell = pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AlexChen/suny_project/ihc_measurements/SUNY_bcell_qupath_measurements.csv')
    df_mac = pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AlexChen/suny_project/ihc_measurements/SUNY_mac_qupath_measurements.csv')
    
    df_cd4 = clean_ihc_data(df_cd4, 'cd4')
    df_cd8 = clean_ihc_data(df_cd8, 'cd8')
    df_bcell = clean_ihc_data(df_bcell, 'bcell')
    df_mac = clean_ihc_data(df_mac, 'mac')

    df_comb = df[['ID', 'T_cells', 'B_cells', 'Macrophages']]
    df_comb = df_comb.merge(df_cd4, how='inner', on='ID')
    df_comb = df_comb.merge(df_cd8, how='inner', on='ID')
    df_comb = df_comb.merge(df_bcell, how='inner', on='ID')
    df_comb = df_comb.merge(df_mac, how='inner', on='ID')
    df_comb['cd4_count'] = df_comb['cd4_count'].fillna(0)
    df_comb['cd8_count'] = df_comb['cd4_count'].fillna(0)
    df_comb['cd4_cd8'] = df_comb.apply(lambda row: row['cd4_count'] + row['cd8_count'], axis=1)
    df_comb = df_comb[df_comb['ID'].str.contains('UR-PDL1-LR-041|UR-PDL1-LR-048|UR-PDL1-LR-083')==False]

    df_comb_bcell = df_comb.dropna(subset='bcell_count')
    
    r_tcell = []
    r_bcell = []
    r_mac = []
    num_bootstrap = 1000
    for i in tqdm(range(num_bootstrap)):
        df_comb_resample = resample(df_comb, random_state=i)

        r = cor(df_comb_resample['cd4_cd8'], df_comb_resample['T_cells'])
        r_tcell.append(r[0])

        r = cor(df_comb_resample['mac_count'], df_comb_resample['Macrophages'])
        r_mac.append(r[0])

    # B cell separate due to nan values
    for i in tqdm(range(num_bootstrap)):
        df_comb_resample = resample(df_comb_bcell, random_state=i)

        assert df_comb_resample['bcell_count'].isna().sum() == 0 
        r = cor(df_comb_resample['bcell_count'], df_comb_resample['B_cells'])
        r_bcell.append(r[0])

    r_dict = {'T cells':r_tcell, 'B cells':r_bcell, 'Macrophages':r_mac} 
    results = {key: {} for key in r_dict.keys()}
    
    for key in r_dict.keys():
        lower_bound1 = round(np.percentile(r_dict[key],2.5),3)
        upper_bound1 = round(np.percentile(r_dict[key],97.5),3)
        print(f'{key} 95% confidence interval:    Scores = [{lower_bound1}, {upper_bound1}] ')
        print(f'{key} Mean:    Scores = {np.mean(r_dict[key])} ')
        #print(f'{key} Median:    Scores Single Task = {np.median(r_single[key])}    |    Scores Multitask = {np.median(r_multi[key])} ')

        plotRoot = f'predictions/histograms/'
        if not os.path.exists(plotRoot):
            os.makedirs(plotRoot)
        plt.hist(r_dict[key])
        plt.savefig(os.path.join(plotRoot, key + '.png'))
        plt.close()

        results[key]['lower_single'] = lower_bound1
        results[key]['upper_single'] = upper_bound1
        results[key]['mean_single'] = np.mean(r_dict[key])

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reset_index().rename(columns={df.index.name:'task'})
    print(df)
    
    df.to_csv(f'predictions/{cor_type}_r_CI_IHC.csv', index=False)


if __name__ == "__main__":
    bootstrap_ihc()

