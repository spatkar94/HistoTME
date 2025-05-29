import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr 
from sklearn.utils import resample
from tqdm import tqdm



def corr_CI(ctype, cor_type = 'pearson'):
    '''
    function to compute pearson correlation between HistoTME predicted signatures and ground truth along with confidence intervals 
    '''
    if cor_type == 'pearson':
        cor = pearsonr
    elif cor_type == 'spearman':
        cor = spearmanr
   
    df_multi_uni = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_uni_5fold.csv',index_col=0)
    df_multi_uni2 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_uni2_5fold.csv',index_col=0)
    df_multi_virchow= pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_virchow_5fold.csv',index_col=0)
    df_multi_virchow2 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_virchow2_5fold.csv',index_col=0)
    df_multi_hoptimus0 = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_hoptimus0_5fold.csv',index_col=0)
    df_multi_gigapath = pd.read_csv(f'predictions/CPTAC_{ctype}_predictions_gigapath_5fold.csv',index_col=0)


    #merged ensemble predictions
    df_multi = pd.concat([df_multi_uni, df_multi_uni2, df_multi_virchow, df_multi_virchow2, df_multi_hoptimus0, df_multi_gigapath]).groupby(level=0).mean()

    #ground truth
    df_gt = pd.read_csv(f'predictions/CPTAC_{ctype}_gt_multitask_uni2_5fold.csv', index_col=0)
    cols = df_gt.columns
    df_multi = df_multi.sort_index()
    df_gt = df_gt.sort_index()
    
    r_multi = {key: [] for key in cols}
    num_bootstrap = 1000
    for i in tqdm(range(num_bootstrap)):
        df_multi_resample = resample(df_multi, random_state=i)
        df_gt_resample = resample(df_gt, random_state=i)

        for col in cols:
            r = cor(df_multi_resample[col], df_gt_resample[col])
            r_multi[col].append(r[0])

    results = {key: {} for key in r_multi.keys()}

    for key in r_multi.keys():
        lower_bound = round(np.percentile(r_multi[key],2.5),3)
        upper_bound = round(np.percentile(r_multi[key],97.5),3)
        print(f'{key} 95% confidence interval: Scores Multitask = [{lower_bound}, {upper_bound}]')
        print(f'{key} Mean: Scores Multitask = {np.mean(r_multi[key])} ')

        results[key]['lower_multi'] = lower_bound
        results[key]['upper_multi'] = upper_bound
        results[key]['mean_multi'] = np.mean(r_multi[key])

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reset_index().rename(columns={df.index.name:'task'})
    print(df)
    
    df.to_csv(f'predictions/{cor_type}_r_CI_{ctype}_ensemble.csv', index=False)


if __name__ == "__main__":
    for ctype in ['LUAD','LSCC','UCEC','HNSCC','CCRCC','PDA','GBM']:
        print(f'Processing results for: {ctype}')
        corr_CI(ctype)
    

