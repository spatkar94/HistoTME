import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from run import set_seed
from data import *
from model import *
from utils import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def predict(epoch, mode, dataloader, model, optimizer):
    predictions = {}
    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            ID = data['ID'][0]
            predictions[ID] = {}
            #labels = data['labels'].to(device)
            # Remember to remove softmax when getting attention map
            A, multitask_slide_preds = model(features)
            
            for key in multitask_slide_preds.keys():
                multitask_pred = multitask_slide_preds[key].squeeze().detach().cpu().numpy()
                predictions[ID][key] = multitask_pred
        
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

def main(args):
    train_dataset, val_dataset, _, feat_dim, multitask_list = load_dataset(args['dataset'], args['embed'])
    
    load_model = args['load']

    if args['cohort'] == 'CPTAC':
        df = val_dataset
    elif args['cohort'] == 'TCGA':
        df = train_dataset

    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1
    loader = build_mil_loader(args, df, "test", None, task_counts)

    model = multitask_ABMIL(task_counts, feat_dim, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", args['embed'], load_model), optimizer)
    model.eval()
    return predict(best_epoch, "test", loader, model, optimizer) 

if __name__ == "__main__":
    set_seed(1)
    #args = {'num_workers':8}

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task type for model to predict")
    parser.add_argument("--cohort", type=str, help="Cohort for prediction")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--embed", default='uni', type=str)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    if args['task'] is None or args['cohort'] is None:
        print('Please include a task type and cohort. For example -> --task=singletask --cohort=CPTAC')
        exit()

    cohort = args['cohort']
    task = args['task']
    
    if task == 'multitask':
        # For multitask
        datasets = ['antitumor', 'protumor', 'cancer', 'angio']
        models = ['abmil_antitumor_huber', 'abmil_protumor_huber', 'abmil_cancer_huber', 'abmil_angio_huber']
    elif task == 'singletask':
        # for singletask
        datasets = ['IFNG', 'MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cell_traffic', 'NK_cells', 'T_cells', 'B_cells', 'M1_signatures', 
                    'Th1_signature', 'Antitumor_cytokines', 'Checkpoint_inhibition', 'Treg', 'T_reg_traffic', 'Neutrophil_signature', 'Granulocyte_traffic',
                    'MDSC', 'MDSC_traffic', 'Macrophages', 'Macrophage_DC_traffic', 'Th2_signature', 'Protumor_cytokines', 'CAF', 'Matrix', 'Matrix_remodeling',
                    'Angiogenesis', 'Endothelium', 'Proliferation_rate', 'EMT_signature']
        models = [ f'abmil_{dataset}_huber' for dataset in datasets ]
    else:
        raise Exception('Please enter a valid task type')

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

    if not os.path.exists('predictions'):
        os.mkdir('predictions/')
    df_new.to_csv(f'predictions/{cohort}_predictions_{task}_UNI.csv', index=False)



