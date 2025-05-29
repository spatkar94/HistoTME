import numpy as np
import pandas as pd
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from run import set_seed
from data import *
from model import *
from utils import *
from scipy.stats import pearsonr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(epoch, mode, dataloader, model):
    '''
    function to generate region/WSI/patient level predictions of signatures
    Input: 
        epoch - best epoch
        mode - test
        dataloader - dataloader holding bag of embeddings for each WSI
        model - pretrained model
    '''
    predictions = {}
    model.eval()
    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            ID = data['ID'][0]
            predictions[ID] = {}
            
            # Remember to remove softmax when getting attention map
            A, multitask_slide_preds = model(features.type(torch.float32), training=False)
            
            for key in multitask_slide_preds.keys():
                multitask_pred = multitask_slide_preds[key].squeeze().detach().cpu().numpy()
                predictions[ID][key] = multitask_pred

            t.update()
        
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

def main(args):
    test_dataset, feat_dim, multitask_list = load_data(signatures=args['dataset'],
                                                             embeddings_folder=args['embeddings_folder'],
                                                             cancer_type=args['cancer_type'])
    
    load_model = args['load']


    task_counts = {}
    
    for task in multitask_list:
        task_counts[task] = 1
    loader = build_mil_loader(args, test_dataset, "test", None, task_counts)

    print(task_counts)
    df_folds = []
    ids = []

    #mean over 5 folds
    for fold in range(5):
        model = multitask_ABMIL(task_counts, feat_dim, 1)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

        best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", str(fold), args['embed'],load_model), optimizer)
        
        df = predict(best_epoch, "test", loader, model)
        ids = df['ID'].values
        df_folds.append(df.iloc[:,1:])

    df_final = pd.concat(df_folds).groupby(level=0).mean()
    df_final.insert(0,'ID',ids)
    df_final['ID'] = df_final['ID'].str.replace(',','_')
    df_final['ID'] = df_final['ID'].str.replace(' ','_')
    return df_final

if __name__ == "__main__":
    set_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_folder", type=str, help="path to embeddings")
    parser.add_argument("--cohort", type=str, help="Cohort name")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--embed", default='gigapath', type=str)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    if args['task'] is None or args['cohort'] is None:
        print('Please include a task type and cohort. For example -> --task=singletask --cohort=CPTAC')
        exit()

    cohort = args['cohort']
    embed = args['embed']

    datasets = ['antitumor', 'protumor', 'cancer', 'angio']
    models = ['abmil_antitumor_huber', 'abmil_protumor_huber', 'abmil_cancer_huber', 'abmil_angio_huber']


    df_all = []
    count = 1

    for dataset, model in zip(datasets, models):
        print(f'Starting on {dataset} {count}/{len(datasets)}')
        args['dataset'] = dataset
        args['load'] = model
        df = main(args)
        df_all.append(df)
        count+=1

    df_new = df_all[0]
    for i in range(len(df_all)-1):
        df_new = pd.merge(df_new, df_all[i+1], on='ID')
    

    #save results to csv file
    if not os.path.exists('predictions'):
        os.mkdir('predictions/')

    df_new.to_csv(f'predictions/{cohort}_predictions_{embed}.csv', index=False)


