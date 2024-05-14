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
import re

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def predict(epoch, mode, dataloader, model, optimizer):
    predictions = {}
    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            ID = data['ID'][0]
            predictions[ID] = {}

            A, multitask_slide_preds = model(features)
            
            for key in multitask_slide_preds.keys():
                multitask_pred = multitask_slide_preds[key].squeeze().detach().cpu().numpy()
                predictions[ID][key] = multitask_pred

            t.update()
        
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

def get_id(x):
    # remove parentheses (a)
    x = re.sub(r"\([^()]*\)", '', x)
    return x[0:14]

def main(args):
    _, _, _, feat_dim, multitask_list = load_dataset(args['dataset'])
    
    dataRoot = '/mnt/synology/ICB_Data_SUNY/UNI_features'
    all_list = os.listdir(dataRoot)
    suny_list = [i for i in all_list if i.startswith('UR-PDL1')]

    suny_dict = {}
    for case in suny_list:
        file_path = os.path.join(dataRoot, case)
        suny_dict[case] = file_path

    df = pd.DataFrame.from_dict(suny_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID', 0:'file_path'}, inplace=True)
    df['ID'] = df['ID'].apply(lambda x: get_id(x))

    #df = df[df['ID']=='UR-PDL1-LR-030']
    
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    test_dataset = pd.merge(df, grouped_paths, on='ID', how='left')
    for task in multitask_list:
        test_dataset[task] = 999
    test_dataset['response_label'] = 999

    load_model = args['load']
    save = load_model + '_regression'
    
    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1
    test_loader = build_mil_loader(args, test_dataset, "test", None, task_counts)

    model = multitask_ABMIL(task_counts, feat_dim, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", load_model), optimizer)
    model.eval()
    return predict(best_epoch, "test", test_loader, model, optimizer) 

if __name__ == "__main__":
    set_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task type for model to predict")
    parser.add_argument("--num_workers", default=8, type=int)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    if args['task'] is None:
        print('Please include a task type. For example -> --task=singletask')
        exit()

    task = args['task']

    if task == 'multitask':
        # For multitask
        datasets = ['antitumor', 'protumor', 'cancer', 'angio']
    elif task == 'singletask':
        # for singletask
        datasets = ['IFNG', 'MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cell_traffic', 'NK_cells', 'T_cells', 'B_cells', 'M1_signatures',
                    'Th1_signature', 'Antitumor_cytokines', 'Checkpoint_inhibition', 'Treg', 'T_reg_traffic', 'Neutrophil_signature', 'Granulocyte_traffic',
                    'MDSC', 'MDSC_traffic', 'Macrophages', 'Macrophage_DC_traffic', 'Th2_signature', 'Protumor_cytokines', 'CAF', 'Matrix', 'Matrix_remodeling',
                    'Angiogenesis', 'Endothelium', 'Proliferation_rate', 'EMT_signature']
    else:
        raise Exception('Please enter a valid task type')
     
    models = [ f'abmil_{dataset}_huber' for dataset in datasets ]

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
    df_new.to_csv(f'predictions/SUNY_predictions_{task}_UNI.csv', index=False)

