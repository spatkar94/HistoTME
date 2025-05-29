import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import timm
from tqdm import tqdm
import h5py
import os
from utils import WSI, Macenko_Normalizer
import argparse
import cv2
from huggingface_hub import login
torch.multiprocessing.set_sharing_strategy('file_system')

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='w'):
    '''
    saves data to h5py format
    :param output_path: path to output h5py file
    :param asset_dict: dictionary containing the data
    :param attr_dict:
    :param mode:
    :return:
    '''
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        #print(key)
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,
                                        dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    print(key)
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
    file.close()
    print("finished writing to hdf5 file...")
    return output_path

def get_embeddings(model, input_img_tensor):
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(input_img_tensor)

    return(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_loc', type=str, default=None, help='path to WSI patches')
    parser.add_argument('--save_loc',type = str, default = None, help='path to location where embeddings will be saved')
    args = parser.parse_args()
    #INPUT SLIDE
    SLIDE_DIR = args.patches_loc
    SAVE_LOC = args.save_loc
    SAVE_NAME = os.path.basename(SLIDE_DIR)
    
    if os.path.exists(os.path.join(SAVE_LOC, SAVE_NAME+'_features.hdf5')):
        print('file exists...')
        print('skipping')
    else:
        #SET UP CUDA DEVICE
        device=torch.device('cuda:3')

        #set up reference image for stain normalization
        target_array = cv2.imread('/mnt/synology/Pan_TCGA_patches/TCGA-T1-A6J8-01Z-00-DX1.013DA5DA-0753-4EBC-99D9-F8817059A202/patches/TCGA-T1-A6J8-01Z-00-DX1.013DA5DA-0753-4EBC-99D9-F8817059A202_43_22.png')
        target_array = cv2.cvtColor(target_array, cv2.COLOR_BGR2RGB)
        
        #Patch Normalization
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        
        hopt_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]
            )

        # Convert target image to torch tensor, and reshape it into (1,3,H,W)
        target_tensor = torch.unsqueeze(hopt_transforms(target_array),0)
        target_tensor = target_tensor.to(device) # optional

        stainnorm = Macenko_Normalizer()
        stainnorm.fit(target_tensor)

        #Initialize Slide Dataset
        slide = WSI(slide_dir=SLIDE_DIR,
                    transforms=hopt_transforms)

        #Initialize Data loader
        slide_loader = DataLoader(slide, batch_size=16, shuffle=False, num_workers=12,drop_last=False)

        #log into huggingface with personal access token

        login('')
        #Initialize uni2 foundation model

        timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }

        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        
        model = model.to(device)
        model.eval()


        #generate embeddings
        wsi_features = []
        wsi_coords = []
        for patches, coords in tqdm(slide_loader):
            with torch.no_grad():
                patches = patches.to(device)
                try:
                    #perform Macenko stain normalization
                    norm_patches = stainnorm.transform(patches)

                    #standardize intensities
                    norm_patches = transforms.Normalize(mean=mean, std=std)(norm_patches)

                    features = get_embeddings(model, norm_patches)
                except Exception as e:
                    print('error: ', e)
                    print('omitting stain normalization')
                    #standardize intensities
                    norm_patches = transforms.Normalize(mean=mean, std=std)(patches)
                    features = get_embeddings(model, norm_patches)
                
                wsi_coords.append(coords.cpu().numpy())
                wsi_features.append(features.cpu().numpy())

        attr_dict = {'features': {'wsi_name': SAVE_NAME +'.svs','label': ''}}

        asset_dict = {'coords': np.vstack(wsi_coords), 'features': np.vstack(wsi_features)}


        #Save embeddings
        if not os.path.exists(SAVE_LOC):
            os.mkdir(SAVE_LOC)

        save_hdf5(os.path.join(SAVE_LOC, SAVE_NAME + '_features.hdf5'),
                                    asset_dict=asset_dict, attr_dict=attr_dict)

