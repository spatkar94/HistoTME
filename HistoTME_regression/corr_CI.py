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

def corr_CI(ctype):
    #df_single = pd.read_csv('predictions/cptac_predictions_singletask_UNI.csv')
    df_multi_uni = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_5fold.csv',index_col=0)
    df_multi_uni2 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni2_5fold.csv',index_col=0)
    df_multi_virchow= pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_virchow_5fold.csv',index_col=0)
    df_multi_virchow2 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_virchow2_5fold.csv',index_col=0)
    df_multi_hoptimus0 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_hoptimus0_5fold.csv',index_col=0)
    df_multi_gigapath = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_gigapath_5fold.csv',index_col=0)

    #df_multi_uni = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_fold0.csv',index_col=0)
    #df_multi_uni2 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_fold1.csv',index_col=0)
    #df_multi_uni3= pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_fold2.csv',index_col=0)
    #df_multi_uni4 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_fold3.csv',index_col=0)
    #df_multi_uni5 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_multitask_uni_fold4.csv',index_col=0)

    df_multi = pd.concat([df_multi_uni, df_multi_uni2, df_multi_virchow, df_multi_virchow2, df_multi_hoptimus0, df_multi_gigapath]).groupby(level=0).mean()
    df_multi = df_multi_uni2

    #csv_path = '/mnt/synology/ICB_Data_SUNY/merged_masterfile_tme_signatures.csv'
    df_gt = pd.read_csv(f'predictions/CPTAC_{ctype}_gt_multitask_uni2_5fold.csv', index_col=0)

    #grouped_paths = df_gt.groupby('ID')['file_path'].apply(list).reset_index()
    #df_gt = df_gt.drop('file_path', axis=1).drop_duplicates()
    #df_gt = pd.merge(df_gt, grouped_paths, on='ID', how='left')
    #df_gt = df_gt[df_gt['ID'].str.startswith('C3')]
    #df_gt = df_gt.drop(columns=['file_path', 'split', 'response_label', 'subtype'])

    cols = df_gt.columns

    #df_single = df_single.sort_values(by=['ID'])
    df_multi = df_multi.sort_index()
    df_gt = df_gt.sort_index()
    
    #r_single = {key: [] for key in cols}
    r_multi = {key: [] for key in cols}
    num_bootstrap = 1000
    for i in tqdm(range(num_bootstrap)):
        #df_single_resample = resample(df_single, random_state=i)
        df_multi_resample = resample(df_multi, random_state=i)
        df_gt_resample = resample(df_gt, random_state=i)

        for col in cols:
            #r = cor(df_single_resample[col], df_gt_resample[col])
            #r_single[col].append(r[0])

            r = cor(df_multi_resample[col], df_gt_resample[col])
            r_multi[col].append(r[0])

    results = {key: {} for key in r_multi.keys()}

    for key in r_multi.keys():
        #lower_bound1 = round(np.percentile(r_single[key],2.5),3)
        #upper_bound1 = round(np.percentile(r_single[key],97.5),3)
        lower_bound2 = round(np.percentile(r_multi[key],2.5),3)
        upper_bound2 = round(np.percentile(r_multi[key],97.5),3)
        print(f'{key} 95% confidence interval: Scores Multitask = [{lower_bound2}, {upper_bound2}]')
        print(f'{key} Mean: Scores Multitask = {np.mean(r_multi[key])} ')


        # Plot histograms of signatures to check for symmetry (assumption of Wilcoxon)
        plotRoot = f'predictions/histograms/'
        if not os.path.exists(plotRoot):
            os.mkdir(plotRoot)
        #plt.hist(r_single[key])
        #plt.savefig(os.path.join(plotRoot, key + '_single.png'))
        #plt.close()

        plt.hist(r_multi[key])
        plt.savefig(os.path.join(plotRoot, key + '_' + ctype + '_multi.png'))
        plt.close()

        #wilcox = wilcoxon(r_single[key], r_multi[key], alternative='less')
        #print(wilcox)
        #print('')
        #results[key]['lower_single'] = lower_bound1
        #results[key]['upper_single'] = upper_bound1
        #results[key]['mean_single'] = np.mean(r_single[key])

        results[key]['lower_multi'] = lower_bound2
        results[key]['upper_multi'] = upper_bound2
        results[key]['mean_multi'] = np.mean(r_multi[key])

        #results[key]['log_p_value'] = -np.log(wilcox[1])

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reset_index().rename(columns={df.index.name:'task'})
    print(df)
    
    df.to_csv(f'predictions/{cor_type}_r_CI_{ctype}_UNI2.csv', index=False)


if __name__ == "__main__":
    for ctype in ['LUAD','LSCC','UCEC','HNSCC','CCRCC','PDA','GBM']:
        print(f'Processing results for: {ctype}')
        corr_CI(ctype)
    

