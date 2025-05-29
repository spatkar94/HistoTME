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
from torchvision import transforms
import timm
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader
import cv2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def predict(epoch, mode, dataloader, model):
    predictions = {}
    model.eval()

    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            features = data['features'].to(device)
            coords = data['coords'].squeeze(0).detach().cpu().numpy()
            


            # Remember to remove softmax when getting attention map
            A, multitask_slide_preds = model(features.type(torch.float32))

                  
            for key in multitask_slide_preds.keys():
                if key not in predictions.keys():
                    predictions[key] = {}

                multitask_pred = multitask_slide_preds[key].squeeze().detach().cpu().numpy()
                
                for i in range(coords.shape[0]):
                    if f'{coords[i][0]}_{coords[i][1]}' not in predictions[key].keys():
                        predictions[key][f'{coords[i][0]}_{coords[i][1]}'] = []

                    predictions[key][f'{coords[i][0]}_{coords[i][1]}'].append(multitask_pred)

            t.update()

    for key in predictions.keys():
        for coords in predictions[key].keys():
            predictions[key][coords] = np.nanmean(predictions[key][coords])

    return predictions


    

def main(args):
    load_model = args['load']

    multitask_list = []
    if args['dataset'] == 'antitumor':
        multitask_list = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
                            
    elif args['dataset'] == 'protumor':
        multitask_list = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
    elif args['dataset'] == 'angio':
        multitask_list = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features
    elif args['dataset'] == 'cancer':
        multitask_list = ['Proliferation_rate', 'EMT_signature']

    feat_dim = None
    if args['embed'] == 'uni':
        feat_dim = 1024
    elif args['embed'] == 'virchow':
        feat_dim = 2560
    elif args['embed'] == 'virchow2':
        feat_dim = 2560
    elif args['embed'] == 'hoptimus0':
        feat_dim = 1536
    elif args['embed'] == 'gigapath':
        feat_dim = 1536
    elif args['embed'] == 'uni2':
        feat_dim = 1536
    task_counts = {}
    for task in multitask_list:
        task_counts[task] = 1
    
    
    dset = SlidingWindow(h5_path=args['h5_path'],window_size=3)
    loader = DataLoader(dset, batch_size=1, num_workers = args['num_workers'],shuffle=False)

    model = multitask_ABMIL(task_counts, feat_dim, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    #val_losses = []
    #for fold in range(5):
    #    best_epoch, _, val_loss = model.load_checkpoint(os.path.join("logs", str(fold), args['embed'],load_model), optimizer)
    #    val_losses.append(val_loss)
    
    #best_fold = np.argmin(val_losses)

    #print(f"Best Fold: {best_fold}")
    #best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", str(best_fold), args['embed'],load_model), optimizer)

    best_epoch, _, _ = model.load_checkpoint(os.path.join("logs", str(args['fold']), args['embed'],load_model), optimizer)

    return predict(best_epoch, "test", loader, model), np.array(dset.coords), np.array(dset.barcodes) 

if __name__ == "__main__":
    set_seed(1)
    #args = {'num_workers':8}

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, help="WSI patch embeddings path for prediction")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--embed", default='virchow', type=str)
    parser.add_argument("--save_loc", default='/home/air/Shared_Drives/MIP_network/MIP/spatkar/HistoTME/spatial_predictions')

    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    if args['h5_path'] is None:
        print('Please include a embeddings path. For example -> --h5_path=/mnt/synology/hest_pancan_st/patches/TENX141.h5')
        exit()

    embed = args['embed']

    
    
    # For multitask
    datasets = ['antitumor', 'protumor', 'cancer', 'angio']
    models = ['abmil_antitumor_huber', 'abmil_protumor_huber', 'abmil_cancer_huber', 'abmil_angio_huber']
    #
    df_folds = []
    for fold in range(5):
        print(f'Fold:{fold}')
        preds_all = {}
        barcodes_all = {}
        coords_all = {}
        gt_all = []
        count = 1

        for dataset, model in zip(datasets, models):
            print(f'Starting on {dataset} {count}/{len(datasets)}')
            args['dataset'] = dataset
            args['load'] = model
            args['fold'] = str(fold)
            preds,coords,barcodes = main(args)
            preds_all[dataset] = preds
            coords_all[dataset] = coords
            barcodes_all[dataset] = barcodes
            count+=1

        df = pd.DataFrame({'x':coords_all[datasets[0]][:,0],'y':coords_all[datasets[0]][:,1]},index=barcodes_all[datasets[0]][:,0])
        for dataset in datasets:
            for signature in preds_all[dataset].keys():
                outputs = []
                for i in range(df.shape[0]):
                    df_row = df.iloc[i,:]
                    coord = f"{int(df_row['x'])}_{int(df_row['y'])}"
                    outputs.append(preds_all[dataset][signature][coord])
                df[signature] = outputs
        print(df)
        df_folds.append(df)

    df_final = pd.concat([df.iloc[:,3:] for df in df_folds]).groupby(level=0).mean()
    df_final = pd.concat([df_folds[0].iloc[:,:3], df_final],axis=1)
    
    if not os.path.exists(f"{args['save_loc']}/{args['embed']}"):
        os.makedirs(f"{args['save_loc']}/{args['embed']}")
    
    print(df_final)
    savename = os.path.basename(args['h5_path']).replace('.h5','_5fold.csv').replace('_features.hdf5','_5fold.csv').replace('h5_features.hdf5','_5fold.csv')
    df_final.to_csv(os.path.join(f"{args['save_loc']}/{args['embed']}",savename),sep=',')
    

