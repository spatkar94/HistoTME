import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import re

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

def load_data_tcga(ctypes, signatures, embeddings_folder, n_splits = 5):
    """
    Load dataframe containing WSI FM embedding file paths and mol signatures from TCGA    
    """
    import glob
    from sklearn.model_selection import train_test_split
    tcga_dataset = pd.DataFrame({})
    tme_signatures = pd.read_csv('/mnt/synology/ICB_Data_SUNY/pantcga_tme_signatures.csv')
    tcga_clindata = pd.read_excel('/mnt/synology/ICB_Data_SUNY/TCGA-CDR-SupplementalTableS1.xlsx',index_col=0)
    tme_signatures.rename(columns={'Unnamed: 0':'bcr_patient_barcode'}, inplace=True)
    embedding_paths = []
    for ext in ['*.hdf5','*.h5']:
        embedding_paths.extend(glob.glob(os.path.join(embeddings_folder,ext)))
    tcga_dataset['bcr_patient_barcode'] = [os.path.basename(x)[:12] for x in embedding_paths]
    tcga_dataset['embedding_paths'] = embedding_paths
    tcga_dataset = pd.merge(tcga_dataset, tcga_clindata, on=['bcr_patient_barcode'])
    tcga_dataset = tcga_dataset[['bcr_patient_barcode','embedding_paths','type']]
    tcga_dataset = pd.merge(tcga_dataset, tme_signatures, on=['bcr_patient_barcode'])

    non_sig_columns = ['bcr_patient_barcode','embedding_paths','type']
    
    pred_tasks = []
    if signatures == 'protumor':
        pred_tasks = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        
    elif signatures == 'antitumor':
        pred_tasks = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
        
    elif signatures == 'angio':
        pred_tasks = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features

    elif signatures == 'cancer':
        pred_tasks = ['Proliferation_rate', 'EMT_signature']
    
    else:
        pred_tasks = signatures
    
    if not isinstance(pred_tasks, list):
        pred_tasks = [pred_tasks]

    if not any(sig in tcga_dataset.columns for sig in pred_tasks):
        raise ValueError('please verify if all signature names are correct.')
    
    if not all(type in tcga_dataset['type'].values for type in ctypes):
        if not ctypes == 'all':
            raise ValueError('please verify if cancer type names are correct.')
    
    df = tcga_dataset[non_sig_columns+pred_tasks]
    df.rename(columns = {'bcr_patient_barcode':'ID','embedding_paths':'file_path'}, inplace=True)
    
    if not ctypes == 'all':
        df = df[df['type'].isin(ctypes)]
    #df.drop('type',axis=1, inplace=True)

    # grouping together patients with multiple slides
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')

    #print(df.columns)
    # Create a KFold object
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Split the DataFrame into 80% train and 20% val

    #df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    train_val_splits = {}
    for fold, (train_index, val_index) in enumerate(kf.split(X = df['file_path'], y=df['type'])):
        # Get the training and testing data for this fold
        train_val_splits[fold] = {}
        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]

        train_val_splits[fold]['train'] = df_train
        train_val_splits[fold]['val'] = df_val

        #print(df_train['type'].value_counts())
        #print(df_val['type'].value_counts())

    #train_val_splits['train'] = df_train
    #train_val_splits['val'] = df_val

    #print(train_val_splits['train']['type'].value_counts().T)
    #print(train_val_splits['val']['type'].value_counts().T)
    
    embedding_dim = 0
    if 'UNI' in embedding_paths[0]:
        embedding_dim = 1024
    elif 'Virchow' in embedding_paths[0]:
        embedding_dim = 2560
    elif 'Hoptimus0' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Gigapath' in embedding_paths[0]:
        embedding_dim = 1536
    
    return train_val_splits, embedding_dim, pred_tasks

def get_id(x):
    # remove parentheses (a)
    x = re.sub(r"\([^()]*\)", '', x)
    return x[0:14]

def get_id2(x):
    # remove suffix
    x = x.split('_')[0]
    return x

def load_data(signatures, embeddings_folder, cancer_type = "NSCLC"):
    import glob
    embedding_paths = []
    for ext in ['*.hdf5','*.h5']:
        embedding_paths.extend(glob.glob(os.path.join(embeddings_folder,ext)))
    pred_tasks = []
    if signatures == 'protumor':
        pred_tasks = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        
    elif signatures == 'antitumor':
        pred_tasks = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
        
    elif signatures == 'angio':
        pred_tasks = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features

    elif signatures == 'cancer':
        pred_tasks = ['Proliferation_rate', 'EMT_signature']
    
    else:
        pred_tasks = signatures

    
    df = pd.DataFrame({'ID':[os.path.basename(x).replace('_features.hdf5','') for x in embedding_paths],
                       'file_path':embedding_paths})
    
    if 'UR-PDL1' in df['ID'].values[0]:
        df['ID'] = df['ID'].apply(lambda x: get_id(x))
    
    #if 'reg' in df['ID'].values[0]:
    #    df['ID'] = df['ID'].apply(lambda x: get_id2(x))
    
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')
    df['type'] = cancer_type

    embedding_dim = 0
    if 'UNI_v1' in embedding_paths[0]:
        embedding_dim = 1024
    elif 'UNI_v2' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'uni_v1' in embedding_paths[0]:
        embedding_dim = 1024
    elif 'uni_v2' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Virchow' in embedding_paths[0] or 'virchow' in embedding_paths[0]:
        embedding_dim = 2560
    elif 'Hoptimus0' in embedding_paths[0] or  'hoptimus0' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Gigapath' in embedding_paths[0] or 'gigapath' in embedding_paths[0]:
        embedding_dim = 1536
    
    return df, embedding_dim, pred_tasks


def load_data_cptac(signatures, embeddings_folder, cancer_type):
    """
    Load dataframe containing WSI FM embedding file paths and mol signatures from TCGA    
    """
    import glob
    cptac_dataset = pd.DataFrame({})
    tme_signatures = pd.read_csv(f'/mnt/synology/ICB_Data_SUNY/cptac_tme_signatures_{cancer_type}.csv')
    #tcga_clindata = pd.read_excel('/mnt/synology/ICB_Data_SUNY/TCGA-CDR-SupplementalTableS1.xlsx',index_col=0)
    tme_signatures.rename(columns={'Unnamed: 0':'bcr_patient_barcode'}, inplace=True)
    embedding_paths = []
    for ext in ['*.hdf5','*.h5']:
        embedding_paths.extend(glob.glob(os.path.join(embeddings_folder,ext)))
    cptac_dataset['bcr_patient_barcode'] = [os.path.basename(x)[:9] for x in embedding_paths]
    cptac_dataset['embedding_paths'] = embedding_paths
    cptac_dataset = pd.merge(cptac_dataset, tme_signatures, on=['bcr_patient_barcode'], how="inner")
    cptac_dataset['type'] = cancer_type
    non_sig_columns = ['bcr_patient_barcode','embedding_paths','type']
    
    pred_tasks = []
    if signatures == 'protumor':
        pred_tasks = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        
    elif signatures == 'antitumor':
        pred_tasks = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
        
    elif signatures == 'angio':
        pred_tasks = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features

    elif signatures == 'cancer':
        pred_tasks = ['Proliferation_rate', 'EMT_signature']
    
    else:
        pred_tasks = signatures

    if not all(sig in cptac_dataset.columns for sig in pred_tasks):
        raise ValueError('please verify if all signature names are correct.')
    
    #if not all(type in cptac_dataset['type'].values for type in ctypes):
    #    if not ctypes == 'all':
    #        raise ValueError('please verify if cancer type names are correct.')
    
    df = cptac_dataset[non_sig_columns+pred_tasks]
    df.rename(columns = {'bcr_patient_barcode':'ID','embedding_paths':'file_path'}, inplace=True)
    
    #if not ctypes == 'all':
    #    df = df[df['type'].isin(ctypes)]
    #df.drop('type',axis=1, inplace=True)

    # grouping together patients with multiple slides
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')


    # Split the DataFrame into 80% train and 20% val
    #train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    embedding_dim = 0
    if 'UNI' in embedding_paths[0]:
        embedding_dim = 1024
        if 'UNI2' in embedding_paths[0]:
            embedding_dim = 1536
    elif 'Virchow' in embedding_paths[0]:
        embedding_dim = 2560
    elif 'Hoptimus0' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Gigapath' in embedding_paths[0]:
        embedding_dim = 1536
    
    return df, embedding_dim, pred_tasks

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
        #label = self.df.iloc[idx]['response_label_Responder']
        #label = torch.from_numpy(np.asarray(label)).float()

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
        #data['labels'] = label
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
        #label = self.df.iloc[idx][self.df.columns.str.startswith('response_label')]
        label = df_row['type']
        
        # getting items for multitask    
        multitask_labels = {}
        for key in self.task_counts.keys():
            try:
                cols = np.array(df_row[key],dtype=float)
                multitask_labels[key] = torch.from_numpy(cols).unsqueeze(dim=0)
            except:
                multitask_labels[key] = 0.0

        # if patient has multiple slides, concatenate all tiles together
        if len(slide_path) > 1:
            features = []
            coords = []
            paths = []
            for path in slide_path:
                with h5py.File(path, 'r') as f:
                    #if 'h5' in path:
                    #    features.append(f['features'][()][0])
                    #    coords.append(f['coords'][()][0])
                    #else:
                    #    features.append(f['features'][()])
                    #    coords.append(f['coords'][()])
                    features.append(f['features'][()])
                    coords.append(f['coords'][()])
                    paths = paths + [path]*f['coords'][()].shape[0]
            ft_np = np.concatenate(features, 0)
            coords_np = np.concatenate(coords, 0)
            slide_path = paths
        elif len(slide_path) == 1:
            with h5py.File(slide_path[0], 'r') as f:
                #if 'h5' in slide_path[0]:
                #    ft_np = f['features'][()][0]
                #    coords_np = f['coords'][()][0]
                #else:
                #    ft_np = f['features'][()]
                #    coords_np = f['coords'][()]
                ft_np = f['features'][()]
                coords_np = f['coords'][()]
        
        #remove na values
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

def create_weighted_sampler(labels):
    values, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(values, counts))
    sample_weights = [1/class_counts[labels[i]] for i in range(len(labels))]
    return WeightedRandomSampler(sample_weights, len(labels), replacement=True)

def build_mil_loader(args, df, subset, bag_size, task_counts):
    try: 
        num_workers = args.num_workers
    except:
        num_workers = args['num_workers']
    
    if not task_counts:
        dataset = milDataset
    else:
        print('using multitask dataset')
        dataset = milMultitaskDataset 

 
    if subset == 'train':
        #print("using weighted random sampler to balance cancer types...")
        #weighted_sampler = create_weighted_sampler(list(df['type'].values))
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=args.batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=1, shuffle=False)
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

class HESTWSI(Dataset):
    def __init__(self, h5_path, transforms = None) -> None:
        self.coords = None
        self.patches = None
        self.barcodes = None
        with h5py.File(h5_path, 'r') as f:
            self.coords = f['coords'][:]
            self.patches = f['img'][:]
            self.barcodes = f['barcode'][:]
        self.barcodes = [str(x).replace("b'","").replace("'","") for x in self.barcodes.squeeze()]
        
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, index):
        patch = self.patches[index]

        if self.transforms is not None:
            try:
                img = self.transforms(patch)
            except Exception as e:
                print('error applying transforms: ', e)
                print('reading image without applying transforms...')
        return img, self.coords[index], self.barcodes[index]

class SlidingWindowHEST(HESTWSI):
    def __init__(self, h5_path, transforms=None, window_size = 10, stride = 1):
        super().__init__(h5_path, transforms)

        xcoords = np.unique(np.sort(self.coords[:,0]))
        ycoords = np.unique(np.sort(self.coords[:,1]))
        patch_size = xcoords[1] - xcoords[0]
        ws = window_size * patch_size
        s = stride * patch_size

        df = pd.DataFrame(self.coords, columns=['xcoord','ycoord'])

        self.windows = {}
        ind = 0
        for x in tqdm(range(0, np.max(xcoords), s)):
            for y in range(0, np.max(ycoords), s):
                window = df[((df['xcoord']>=x) & (df['xcoord']<(x+ws))) &
                        ((df['ycoord']>=y) & (df['ycoord']<(y+ws)))]
                if window.shape[0] > ((window_size*window_size)//3):
                    #get patches
                    self.windows[ind] = list(window.index)
                    ind += 1

    def __len__(self) -> int:
        return len(self.windows)
    def __getitem__(self, index):
        coords_w = self.coords[self.windows[index],:]
        barcodes_w = []
        patches_w = []

        for ind in self.windows[index]:
            patch = self.patches[ind]

            if self.transforms is not None:
                try:
                    patch = self.transforms(patch)
                except Exception as e:
                    print('error applying transforms: ', e)
                    print('reading image without applying transforms...')

            patches_w.append(patch)
            barcodes_w.append(self.barcodes[ind])
        
        patches_w = np.stack(patches_w)
        data = {}
        data['patches'] = patches_w
        data['coords'] = coords_w
        data['barcodes'] = barcodes_w
        return data
    
class SlidingWindow(Dataset):
    def __init__(self, h5_path, window_size = 10, stride = 1):
        with h5py.File(h5_path, 'r') as f:
            self.coords = f['coords'][:]
            self.features = f['features'][:]
            try:
                self.barcodes = f['barcodes'][:].astype(str)
            except:
                self.barcodes = np.array([str(i) for i in range(len(self.coords))]).reshape(-1,1)
                print(self.barcodes.shape)

        df = pd.DataFrame(self.coords, columns=['x','y'])
        sorted_df = df.sort_values(by=['x','y'], ascending=True)
        xcoords = np.unique(sorted_df['x'].values)
        ycoords = np.unique(sorted_df['y'].values)
        wmap = {}
        for i in range(sorted_df.shape[0]):
            wmap[(sorted_df.iloc[i,0],sorted_df.iloc[i,1])] = sorted_df.index[i]

        self.windows = {}
        ind = 0
        for i in tqdm(range(0,len(xcoords)-window_size +1,stride)):
            for j in range(0,len(ycoords)-window_size +1,stride):
                indices = []
                count = 0
                for k in range(i,min(i+window_size, len(xcoords))):
                    for l in range(j,min(j+window_size, len(ycoords))):
                        try:
                            indices.append(wmap[(xcoords[k],ycoords[l])])
                        except:
                            pass
                        count+=1
                if len(indices) > 0:
                    self.windows[ind] = indices
                    ind += 1

    def __len__(self) -> int:
        return len(self.windows)
    def __getitem__(self, index):
        coords_w = self.coords[self.windows[index],:]
        features_w = self.features[self.windows[index],:]
        data = {}
        data['features'] = features_w
        data['coords'] = coords_w
        return data



    
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



