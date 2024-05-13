import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr, spearmanr 
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt

cor_type = 'pearson'
if cor_type == 'pearson':
    cor = pearsonr
elif cor_type == 'spearman':
    cor = spearmanr

def corr_CI():
    df_single = pd.read_csv('predictions/cptac_predictions_singletask_UNI.csv')
    df_multi = pd.read_csv('predictions/cptac_predictions_multitask_UNI.csv')

    csv_path = '/mnt/synology/ICB_Data_SUNY/merged_masterfile_tme_signatures.csv'
    df_gt = pd.read_csv(csv_path)

    grouped_paths = df_gt.groupby('ID')['file_path'].apply(list).reset_index()
    df_gt = df_gt.drop('file_path', axis=1).drop_duplicates()
    df_gt = pd.merge(df_gt, grouped_paths, on='ID', how='left')
    df_gt = df_gt[df_gt['ID'].str.startswith('C3')]
    df_gt = df_gt.drop(columns=['file_path', 'split', 'response_label', 'subtype'])

    cols = df_gt.columns.drop(['ID'])

    df_single = df_single.sort_values(by=['ID'])
    df_multi = df_multi.sort_values(by=['ID'])
    df_gt = df_gt.sort_values(by=['ID'])
    
    r_single = {key: [] for key in cols}
    r_multi = {key: [] for key in cols}
    num_bootstrap = 1000
    for i in tqdm(range(num_bootstrap)):
        df_single_resample = resample(df_single, random_state=i)
        df_multi_resample = resample(df_multi, random_state=i)
        df_gt_resample = resample(df_gt, random_state=i)

        for col in cols:
            r = cor(df_single_resample[col], df_gt_resample[col])
            r_single[col].append(r[0])

            r = cor(df_multi_resample[col], df_gt_resample[col])
            r_multi[col].append(r[0])

    results = {key: {} for key in r_single.keys()}

    for key in r_single.keys():
        lower_bound1 = round(np.percentile(r_single[key],2.5),3)
        upper_bound1 = round(np.percentile(r_single[key],97.5),3)
        lower_bound2 = round(np.percentile(r_multi[key],2.5),3)
        upper_bound2 = round(np.percentile(r_multi[key],97.5),3)
        print(f'{key} 95% confidence interval:    Scores Single task = [{lower_bound1}, {upper_bound1}]    |    Scores Multitask = [{lower_bound2}, {upper_bound2}]')
        print(f'{key} Mean:    Scores Single Task = {np.mean(r_single[key])}    |    Scores Multitask = {np.mean(r_multi[key])} ')


        # Plot histograms of signatures to check for symmetry (assumption of Wilcoxon)
        plotRoot = f'predictions/histograms/'
        if not os.path.exists(plotRoot):
            os.mkdir(plotRoot)
        plt.hist(r_single[key])
        plt.savefig(os.path.join(plotRoot, key + '_single.png'))
        plt.close()

        plt.hist(r_multi[key])
        plt.savefig(os.path.join(plotRoot, key + '_multi.png'))
        plt.close()

        wilcox = wilcoxon(r_single[key], r_multi[key], alternative='less')
        print(wilcox)
        print('')
        results[key]['lower_single'] = lower_bound1
        results[key]['upper_single'] = upper_bound1
        results[key]['mean_single'] = np.mean(r_single[key])

        results[key]['lower_multi'] = lower_bound2
        results[key]['upper_multi'] = upper_bound2
        results[key]['mean_multi'] = np.mean(r_multi[key])

        results[key]['log_p_value'] = -np.log(wilcox[1])

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reset_index().rename(columns={df.index.name:'task'})
    print(df)
    
    df.to_csv(f'predictions/{cor_type}_r_CI_UNI.csv', index=False)


if __name__ == "__main__":
    corr_CI()

