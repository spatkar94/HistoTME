import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py

GROUND_TRUTH_PATH = '/mnt/synology/ICB_Data_SUNY/merged_masterfile_tme_signatures.csv'

def get_source(row): 
    if os.path.basename(row['file_path']).startswith('UR-PDL1'):
        return 'SUNY'
    elif os.path.basename(row['file_path']).startswith('TCGA'):
        return 'TCGA'
    elif os.path.basename(row['file_path']).startswith('C3'):
        return 'CPTAC'
def get_split(row):
    if get_source(row)=='TCGA':
        return 'TRAIN'
    elif get_source(row)=='CPTAC':
        return 'VAL'
    else:
        return 'TEST'

def load_dataset(name, embed_type):
    """
    Load a dataframe containing file paths for a dataset    
    """
    non_feature_columns = ['ID', 'file_path', 'response_label','split']
    csv_path = GROUND_TRUTH_PATH

    df = pd.read_csv(csv_path)
    if embed_type.lower() == 'ctranspath':
        pass
    elif embed_type.lower() == 'retccl':
        df['file_path'] = df['file_path'].apply(lambda x: x.replace('transpath_features', 'retccl_features'))
    elif embed_type.lower() == 'uni':
        df['file_path'] = df['file_path'].apply(lambda x: x.replace('transpath_features', 'UNI_features'))
    else:
        raise Exception(f"{embed_type} is not a valid embedding. Please select from ctranspath, retccl, or uni...")

    if name == "tme": 
        pass
    elif name == 'suny':
        df['source'] = df.apply(lambda row: get_source(row), axis=1)
        df = df[df['source'] == 'SUNY']
        df = df[non_feature_columns]
    elif name == 'tme_ft_pred':
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
    elif name == 'ifng':
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[non_feature_columns]
    elif name == 'protumor':
        features = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[features + non_feature_columns]
    elif name == 'antitumor':
        features = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines', 'IFNG'] # 12 features
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[features + non_feature_columns]
    elif name == 'angio':
        features = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[features + non_feature_columns]
    elif name == 'cancer':
        features = ['Proliferation_rate', 'EMT_signature']
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[features + non_feature_columns]
    elif name == 'macrophage':
        features = ['Macrophage_DC_traffic', 'Macrophages', 'M1_signatures', 'MHCII', 'IFNG']    
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[features + non_feature_columns]
    elif name in df.columns:
        df['split'] = df.apply(lambda row: get_split(row), axis=1)
        df = df[non_feature_columns + [name]]
    else:
        print('Please choose a valid name. Exiting...')
        exit()

    multitasks = df.drop(columns=non_feature_columns).columns.tolist()

    df_dummy = pd.get_dummies(df, columns=['response_label'], dtype=int)
    df_dummy.loc[df.response_label.isnull(), df_dummy.columns.str.startswith('response_label')] = -999
    df = df_dummy

    # grouping together patients with multiple slides
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')

    train_tiles = df[df['split']=='TRAIN']
    val_tiles = df[df['split']=='VAL']
    test_tiles = df[df['split']=='TEST']

    print(f"length of train tiles = {len(train_tiles)} | length of validation tiles = {len(val_tiles)} | length of test tiles = {len(test_tiles)}")
    
    slide_path = df.iloc[0]['file_path'][0]
    with h5py.File(slide_path, 'r') as f: # coords, features
        #coords = f['coords'][()]
        features = f['features'][()]

    feat_dim=features.shape[1]  # shape of pre-trained embeddings
    return train_tiles, val_tiles, test_tiles, feat_dim, multitasks

class milDataset(Dataset):
    '''
    Dataset used for attention-based MIL learning (classification)
    '''
    def __init__(self, df, task_counts=None, bag_size=None):
        super(milDataset, self).__init__()
        self.df = df
        self.bag_size = bag_size

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        ID = df_row['ID']
        slide_path = df_row['file_path']
        label = self.df.iloc[idx]['response_label_Responder']
        label = torch.from_numpy(np.asarray(label)).float()

        # if patient has multiple slides, concatenate all tiles together
        if len(slide_path) > 1:
            features = []
            coords = []
            paths = []
            for path in slide_path:
                with h5py.File(path, 'r') as f:
                    features.append(f['features'][()])
                    coords.append(f['coords'][()])
                    paths = paths + [path]*f['coords'][()].shape[0]
            ft_np = np.concatenate(features, 0)
            coords_np = np.concatenate(coords, 0)
            slide_path = paths
        elif len(slide_path) == 1:
            with h5py.File(slide_path[0], 'r') as f:
                ft_np = f['features'][()]
                coords_np = f['coords'][()]

        #remove na values in TCGA-60-2710
        if np.isnan(ft_np).any():
            ft_np = ft_np[~np.isnan(ft_np).any(axis=1)]
        
        ft_pt = torch.from_numpy(ft_np)
   
        if self.bag_size:
            ft_pt, ft_len = _to_fixed_size_bag(ft_pt, bag_size=self.bag_size) 
        else:
            ft_len = len(ft_pt)

        data = {}          
        data['features'] = ft_pt
        assert not ft_pt.isnan().any(), slide_path
        data['ft_lengths'] = torch.from_numpy(np.asarray(ft_len))
        data['labels'] = label
        data['slide_path'] = slide_path

        return data

    def __len__(self):
        return len(self.df)

class milMultitaskDataset(Dataset):
    '''
    Dataset used for attention-based MIL learning (multitask classification)
    '''
    def __init__(self, df, task_counts, bag_size=None):
        super(milMultitaskDataset, self).__init__()
        self.df = df
        self.task_counts = task_counts
        self.bag_size = bag_size

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        ID = df_row['ID']
        slide_path = df_row['file_path']
        label = self.df.iloc[idx][self.df.columns.str.startswith('response_label')]
        label = torch.from_numpy(label.to_numpy().astype(float))
        
        # getting items for multitask    
        multitask_labels = {}
        for key in self.task_counts.keys():
            cols = np.array(df_row[key],dtype=float)
            #cols = df_row.values[col_ind:col_ind+3]
            multitask_labels[key] = torch.from_numpy(cols).unsqueeze(dim=0)

        # if patient has multiple slides, concatenate all tiles together
        if len(slide_path) > 1:
            features = []
            coords = []
            paths = []
            for path in slide_path:
                with h5py.File(path, 'r') as f:
                    features.append(f['features'][()])
                    coords.append(f['coords'][()])
                    paths = paths + [path]*f['coords'][()].shape[0]
            ft_np = np.concatenate(features, 0)
            coords_np = np.concatenate(coords, 0)
            slide_path = paths
        elif len(slide_path) == 1:
            with h5py.File(slide_path[0], 'r') as f:
                ft_np = f['features'][()]
                coords_np = f['coords'][()]

        #remove na values in TCGA-60-2710
        if np.isnan(ft_np).any():
            ft_np = ft_np[~np.isnan(ft_np).any(axis=1)]
        
        ft_pt = torch.from_numpy(ft_np)
   
        if self.bag_size:
            ft_pt, ft_len = _to_fixed_size_bag(ft_pt, bag_size=self.bag_size) 
        else:
            ft_len = len(ft_pt)

        data = {}          
        data['features'] = ft_pt
        data['coords'] = coords_np
        assert not ft_pt.isnan().any(), slide_path
        data['ft_lengths'] = torch.from_numpy(np.asarray(ft_len))
        data['labels'] = label
        data['multitask_labels'] = multitask_labels
        data['slide_path'] = slide_path
        data['ID'] = ID
    
        return data

    def __len__(self):
        return len(self.df)

def build_mil_loader(args, df, subset, bag_size, task_counts):
    shuffle = (subset != "test")
    try: 
        num_workers = args.num_workers
    except:
        num_workers = args['num_workers']
    
    if not task_counts:
        dataset = milDataset
    else:
        print('using multitask dataset')
        dataset = milMultitaskDataset 

    if subset == 'test':
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=1, shuffle=shuffle)
    else:
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=args.batch_size, shuffle=shuffle)
    return loader     

def _to_fixed_size_bag(bag, bag_size):
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat(
        (
            bag_samples,
            torch.zeros(bag_size - bag_samples.shape[0], bag_samples.shape[1]),
        )
    )
    return zero_padded, min(bag_size, len(bag))
    
if __name__ == "__main__":
    df, test_tiles, feat_dim, multitasks = load_dataset('tme')

    def get_source(row): 
        if os.path.basename(row['file_path']).startswith('UR-PDL1'):
            return 'SUNY'
        elif os.path.basename(row['file_path']).startswith('TCGA'):
            return 'TCGA'
        elif os.path.basename(row['file_path']).startswith('C3'):
            return 'CPTAC'

    df['source'] = df.apply(lambda row: get_source(row), axis=1)
    print(df)
    print(df['source'].value_counts())
    print(len(pd.unique(df[df['source']=='CPTAC']['ID'])))
    print(len(pd.unique(df[df['source']=='TCGA']['ID'])))



