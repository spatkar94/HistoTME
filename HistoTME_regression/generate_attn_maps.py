import os
import h5py
import argparse
import openslide
import numpy as np
from PIL import Image
from matplotlib import cm
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svs_loc', type=str, default=None, help='path to svs files')
    parser.add_argument('--attn_maps_loc',type = str, default = None, help='path to attention maps')
    parser.add_argument('--patch_size',type = int, default= 512, help='patch size in pixels')
    parser.add_argument('--mag', type = int, default=20, help='selected magnification')
    parser.add_argument('--out_dir',type=str, help='path to output directory where attention maps are saved')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.attn_maps_loc, '*.hdf5'))
    for file in files:
        try:
            with h5py.File(file) as f:
                attention = f['attention'][:]
                coords = f['coords'][:]

            oslide = openslide.open_slide(os.path.join(args.svs_loc,os.path.basename(file).split('_')[0]))
            acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
            patch_size = round(args.patch_size * acq_mag / args.mag)

            ncol = int(oslide.level_dimensions[0][0]/patch_size)+1
            nrow = int(oslide.level_dimensions[0][1]/patch_size)+1

            ATTENTION = np.zeros((nrow, ncol))
            for i in range(len(coords)):
                ATTENTION[int(coords[i,1]/patch_size), int(coords[i,0]/patch_size)] = (attention[i] - np.min(attention))/(np.max(attention) - np.min(attention))

            attn_map = Image.fromarray(np.uint8(cm.plasma(ATTENTION)*255))

            save_name = os.path.basename(file).split('_')[0].replace('.svs',f'_{file.split("/")[5].split("_")[1]}.png')
            attn_map.save(os.path.join(args.out_dir, save_name))

            print('Done!')
        except:
            print('error occured... moving to next file')


