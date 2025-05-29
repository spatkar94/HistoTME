import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import re

def load_data_tcga(ctypes, signatures, embeddings_folder, n_splits = 5):
    """
    Create master dataframe containing WSI FM embedding file paths and mol signatures from TCGA.
    Returns stratified K-fold CV splits    
    """
    import glob
    from sklearn.model_selection import train_test_split
    tcga_dataset = pd.DataFrame({})
    tme_signatures = pd.read_csv('../example_data/pantcga_tme_signatures.csv')
    tcga_clindata = pd.read_excel('../example_data/TCGA-CDR-SupplementalTableS1.xlsx',index_col=0)
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

    # grouping together patients with multiple slides
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')

    # Create a KFold object
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_val_splits = {}
    for fold, (train_index, val_index) in enumerate(kf.split(X = df['file_path'], y=df['type'])):
        # Get the training and testing data for this fold
        train_val_splits[fold] = {}
        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]

        train_val_splits[fold]['train'] = df_train
        train_val_splits[fold]['val'] = df_val

    
    embedding_dim = 0
    if 'UNI_v1' in embedding_paths[0]:
        embedding_dim = 1024
    elif 'UNI_v2' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Virchow' in embedding_paths[0]:
        embedding_dim = 2560
    elif 'Hoptimus0' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Gigapath' in embedding_paths[0]:
        embedding_dim = 1536
    
    return train_val_splits, embedding_dim, pred_tasks

def load_data_cptac(signature_group, embeddings_folder, cancer_type):
    """
    Load dataframe containing WSI FM embedding file paths and mol signatures from CPTAC   
    """
    cptac_dataset = pd.DataFrame({})
    tme_signatures = pd.read_csv(f'/mnt/synology/ICB_Data_SUNY/cptac_tme_signatures_{cancer_type}.csv')
    
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
    if signature_group == 'protumor':
        pred_tasks = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        
    elif signature_group == 'antitumor':
        pred_tasks = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
        
    elif signature_group == 'angio':
        pred_tasks = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features

    elif signature_group == 'cancer':
        pred_tasks = ['Proliferation_rate', 'EMT_signature']
    
    else:
        #custom signature group
        pred_tasks = signature_group

    if not all(sig in cptac_dataset.columns for sig in pred_tasks):
        raise ValueError('please verify if all signature names are correct.')
    
    
    df = cptac_dataset[non_sig_columns+pred_tasks]
    df.rename(columns = {'bcr_patient_barcode':'ID','embedding_paths':'file_path'}, inplace=True)
    

    # grouping together patients with multiple slides
    grouped_paths = df.groupby('ID')['file_path'].apply(list).reset_index()
    df = df.drop('file_path', axis=1).drop_duplicates()
    df = pd.merge(df, grouped_paths, on='ID', how='left')


    embedding_dim = 0
    if 'UNI_v1' in embedding_paths[0]:
        embedding_dim = 1024
    elif 'UNI_v2' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Virchow' in embedding_paths[0]:
        embedding_dim = 2560
    elif 'Hoptimus0' in embedding_paths[0]:
        embedding_dim = 1536
    elif 'Gigapath' in embedding_paths[0]:
        embedding_dim = 1536
    
    return df, embedding_dim, pred_tasks



def load_data(signature_group, embeddings_folder, cancer_type):
    """
    Load master dataframe containing WSI FM embedding file paths from a test cohort  
    """
    # utility function to extract patient ids from specific insititutional test cohorts. 
    # You can modify these based on cohort-specific naming convention
    def get_pt_id(x):
        # remove parentheses (a)
        x = re.sub(r"\([^()]*\)", '', x)
        return x[0:14]

    embedding_paths = []
    for ext in ['*.hdf5','*.h5']:
        embedding_paths.extend(glob.glob(os.path.join(embeddings_folder,ext)))
    pred_tasks = []
    if signature_group == 'protumor':
        pred_tasks = ['Checkpoint_inhibition', 'Macrophage_DC_traffic', 'T_reg_traffic', 'Treg', 
                    'Th2_signature', 'Macrophages', 'Neutrophil_signature', 'Granulocyte_traffic', 
                    'MDSC_traffic', 'MDSC', 'Protumor_cytokines'] # 11 features
        
    elif signature_group == 'antitumor':
        pred_tasks = ['MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells', 'T_cells', 
                    'T_cell_traffic', 'NK_cells', 'B_cells', 'M1_signatures', 'Th1_signature', 
                    'Antitumor_cytokines'] # 12 features
        
    elif signature_group == 'angio':
        pred_tasks = ['Matrix', 'Matrix_remodeling', 'Endothelium', 'CAF', 'Angiogenesis'] # 5 features

    elif signature_group == 'cancer':
        pred_tasks = ['Proliferation_rate', 'EMT_signature']
    
    else:
        #if a custom group of signatures is provided
        pred_tasks = signature_group

    
    df = pd.DataFrame({'ID':[os.path.basename(x).replace('_features.hdf5','').replace('.h5','') for x in embedding_paths],
                       'file_path':embedding_paths})
    
    if 'UR-PDL1' in df['ID'].values[0]:
        df['ID'] = df['ID'].apply(lambda x: get_pt_id(x))
    
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

class milDataset(Dataset):
    '''
    Dataset class used for setting attention-based MIL learning and inference
    '''
    def __init__(self, df, task_counts=None, bag_size=None):
        super(milDataset, self).__init__()
        self.df = df
        self.bag_size = bag_size

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        ID = df_row['ID']
        slide_path = df_row['file_path']
        

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
        assert not ft_pt.isnan().any(), slide_path
        data['ft_lengths'] = torch.from_numpy(np.asarray(ft_len))
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
        label = df_row['type']
        
        # getting ground truth signature scores for group of signatures   
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
                    #assuming features are of dim nxd, where n is number of tiles, d is embedding dim
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
    '''
    helper function to oversample under-represented/minority classes
    '''
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
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=args.batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset(df, task_counts, bag_size),
                         num_workers=num_workers, batch_size=1, shuffle=False)
    return loader     

def _to_fixed_size_bag(bag, bag_size):
    '''
    helper function to sample a fixed size of tiles from WSI (if bag_size is provided) 
    '''
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
    '''
    base Dataset class to facilitate ABMIL inference on HEST1K data 
    '''
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
    '''
    class to run sliding window ABMIL inference on HEST1K WSI images.
    Creates a dataset object that generates a sliding window instance of tiles within a predefined window of dimensions (window_size x window_sixe)

    '''
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
    '''
    class to run sliding window ABMIL inference on custom WSI images.
    Creates a dataset object that generates a sliding window instance of tiles within a predefined window of dimensions (window_size x window_sixe)
    '''
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

