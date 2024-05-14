import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from run import set_seed
from data import *
from model import *
from utils import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def make_attention_map(epoch, mode, dataloader, model, optimizer, outRoot):

    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            coords = data['coords']
            paths = data['slide_path']
            paths_np = np.array(paths).squeeze()
            paths_unique = np.unique(paths_np)

            A, _ = model(features)

            A = A.squeeze().cpu().detach().numpy()
            coords = coords.squeeze().cpu().detach().numpy()
            for path in paths_unique:
                slide = os.path.basename(path)
                if slide.startswith('UR-PDL1-LB'):
                    continue
                if os.path.exists(os.path.join(outRoot, slide)):
                    print(f" ### {os.path.join(outRoot, slide)} Exists! ###")
                    continue

                #print(f'Saving attention for {os.path.join(outRoot, slide)}')
                A_slide = A[paths_np==path]
                coords_slide = coords[paths_np==path]
                    
                with h5py.File(os.path.join(outRoot, slide), 'w') as f:
                    f.create_dataset('attention', data=A_slide)
                    f.create_dataset('coords', data=coords_slide)

            t.update()

def main(args):
    _,_, _, feat_dim, multitask_list = load_dataset(args['dataset'], embed_type='uni')

    dataRoot = '/mnt/synology/ICB_Data_SUNY/UNI_features'
    all_list = os.listdir(dataRoot)
    suny_list = [i for i in all_list if i.startswith('UR-PDL1')]
    suny_dict = {}
    for case in suny_list:
        file_path = os.path.join(dataRoot, case)
        ID = case[0:14]
        suny_dict[case] = file_path
    df = pd.DataFrame.from_dict(suny_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID', 0:'file_path'}, inplace=True)
    df['ID'] = df['ID'].apply(lambda x: x[0:14])
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    test_dataset = pd.merge(df, grouped_paths, on='ID', how='left')

    for task in multitask_list:
        test_dataset[task] = 999
    test_dataset['response_label'] = 999
    
    load_model = args['load']
    save = load_model + '_regression'
    outRoot = f'/mnt/synology/ICB_Data_SUNY/attention_maps_test/{save}'
    
    if not os.path.exists(outRoot):
        os.makedirs(outRoot)

    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1
    test_loader = build_mil_loader(args, test_dataset, "test", None, task_counts)

    model = ABMIL_attention_map(task_counts, feat_dim, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", 'uni', load_model), optimizer)
    model.eval()
    make_attention_map(best_epoch, "test", test_loader, model, optimizer, outRoot) 

if __name__ == "__main__":
    set_seed(1)
    args = {'num_workers':8}
    datasets = ['antitumor', 'protumor', 'cancer', 'angio']
    models = [ f'abmil_{dataset}_huber' for dataset in datasets ]
    for dataset, model in zip(datasets, models):
        args['dataset'] = dataset
        args['load'] = model
        main(args)



