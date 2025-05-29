import numpy as np
import pandas as pd
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from run import set_seed
from src.data import *
from src.model import *
from src.utils import *
from scipy.stats import pearsonr

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def predict(epoch, mode, dataloader, model):
    predictions = {}
    ground_truth = {}
    model.eval()
    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            ID = data['ID'][0]
            predictions[ID] = {}
            ground_truth[ID] = {}
            #labels = data['labels'].to(device)
            # Remember to remove softmax when getting attention map
            A, multitask_slide_preds = model(features.type(torch.float32), training=False)
            
            for key in multitask_slide_preds.keys():
                multitask_pred = multitask_slide_preds[key].squeeze().detach().cpu().numpy()
                multitask_label = data['multitask_labels'][key].float().squeeze().detach().cpu().numpy()
                predictions[ID][key] = multitask_pred
                ground_truth[ID][key] = multitask_label

            t.update()
        
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)

    gt = pd.DataFrame.from_dict(ground_truth, orient='index')
    gt.reset_index(inplace=True)
    gt.rename(columns={'index':'ID'}, inplace=True)
    return df, gt

def main(args):
    embeddings_folder = {}
    embeddings_folder['uni'] = '/mnt/synology/UNI_Pan_CPTAC'
    embeddings_folder['uni2'] = '/mnt/synology/UNI2_Pan_CPTAC'
    embeddings_folder['virchow'] = '/mnt/synology/Virchow_Pan_CPTAC'
    embeddings_folder['virchow2'] = '/mnt/synology/Virchow2_Pan_CPTAC'
    embeddings_folder['gigapath'] = '/mnt/synology/Gigapath_Pan_CPTAC'
    embeddings_folder['hoptimus0'] = '/mnt/synology/Hoptimus0_Pan_CPTAC'
    test_dataset, feat_dim, multitask_list = load_data_cptac(signatures=args['dataset'],
                                                             embeddings_folder=embeddings_folder[args['embed']],
                                                             cancer_type=args['cancer_type'])
    
    load_model = args['load']


    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1
    loader = build_mil_loader(args, test_dataset, "test", None, task_counts)

    print(task_counts)
    df_folds = []
    gt_folds = []
    ids = []

    #mean over 5 folds
    for fold in range(5):
        model = multitask_ABMIL(task_counts, feat_dim, 1)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

        best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", str(fold), args['embed'],load_model), optimizer)
    
        df,gt = predict(best_epoch, "test", loader, model)
        ids = df['ID'].values
        df_folds.append(df.iloc[:,1:])
        gt_folds = gt 
    
    df_final = pd.concat(df_folds).groupby(level=0).mean()
    df_final['ID'] = ids
    gt_final = gt_folds
    df_final = df_final[gt_final.columns]
    return df_final, gt_final

    


if __name__ == "__main__":
    set_seed(1)
    #args = {'num_workers':8}

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=str, help="Cohort for prediction")
    parser.add_argument("--cancer_type", type=str, help="Cancer type to evaluate")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--embed", default='gigapath', type=str)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    if args['cohort'] is None:
        print('Please include a cohort name. For example -> --cohort=CPTAC')
        exit()

    cohort = args['cohort']
    cancer_type = args['cancer_type']
    embed = args['embed']

    

    # For multitask
    datasets = ['antitumor', 'protumor', 'cancer', 'angio']
    models = ['abmil_antitumor_huber', 'abmil_protumor_huber', 'abmil_cancer_huber', 'abmil_angio_huber']
    

    df_all = []
    gt_all = []
    count = 1

    for dataset, model in zip(datasets, models):
        print(f'Starting on {dataset} {count}/{len(datasets)}')
        args['dataset'] = dataset
        args['load'] = model
        df, gt = main(args)
        df_all.append(df)
        gt_all.append(gt)
        count+=1

    df_new = df_all[0]
    for i in range(len(df_all)-1):
        df_new = pd.merge(df_new, df_all[i+1], on='ID')
    
    gt_new = gt_all[0]
    for i in range(len(gt_all)-1):
        gt_new = pd.merge(gt_new, gt_all[i+1], on='ID')

    for key in df_new.columns[1:]:

        multitask_pred_all = df_new[key].values
        multitask_truth_all = gt_new[key].values
        r = pearsonr(multitask_pred_all, multitask_truth_all)
        mse = np.mean((multitask_pred_all - multitask_truth_all)**2)
        rmse = np.sqrt(mse)
        print(f'{key} rmse = {rmse} | pearson r = {r}')

    if not os.path.exists('predictions'):
        os.mkdir('predictions/')
    df_new.to_csv(f'predictions/{cohort}_{cancer_type}_predictions_{embed}_5fold.csv', index=False)
    gt_new.to_csv(f'predictions/{cohort}_{cancer_type}_gt_{embed}_5fold.csv', index=False)



