import os
import glob
import numpy as np
import openslide
import geojson
import argparse
import torch
import timm
import cv2
import h5py
import pyvips
from tqdm import tqdm
from PIL import Image
from shapely.geometry import MultiPoint, shape, Polygon
from shapely.ops import unary_union
from monai.apps.pathology.transforms import NormalizeHEStains
from wsi_tile_cleanup import filters, utils
from openslide.deepzoom import DeepZoomGenerator
from torchvision import transforms


class WSI_Analyzer:
    def __init__(self, file_location, image_file, save_location, save_name, xml_file, pt_label, patch_size, mag_extract):
        self.file_location = file_location  # path to directory containing svs files
        self.image_file = image_file  # svs/ndpi/czi/qptiff file
        self.xml_file = xml_file  #if you dont have an xml file specify 'none'
        self.save_location = save_location # path to directory where WSI embeddings will be stored
        self.save_name = save_name  # name of the output file holding WSI embeddings
        self.mag_extract = mag_extract  # specify which magnification you wish to pull images from - ONLY SUPPORTS SINGLE MAG
        self.patch_size = patch_size  # specify image size to be saved at specified magnification
        self.pixel_overlap = 0  # specify the level of pixel overlap in your saved images
        self.pt_label = pt_label # specify pt level label if any. Default is 'unlabeled'
        self.limit_bounds = True #do not change this

    def whitespace_check(self, im):
        '''
        checks amount of whitespace in image
        :param im:
        :return: float: proportion of image covered in whitespace
        '''
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw = bw / 255
        prop_ws = (bw > 0.8).sum() / (bw > 0).sum()
        return prop_ws

    def mask2polygons(self, mask, scale):
        '''
        converts binary tissue segmentation mask to ROI
        :param mask: binary mask image
        :param scale: factor by which to scale coordinates (to match base magnification level)
        :return: Polygon representing masked region of interest
        '''
        # find contours from binary mask
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # create polygons from cotours
        polygons = []
        for contour in tqdm(contours):
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = np.round(cvals[i][0] * scale[0], 2)
                cvals[i][1] = np.round(cvals[i][1] * scale[1], 2)
            try:
                if cvals.shape[0] > 2:
                    poly = Polygon(cvals).buffer(0)
                    if poly.area > 100000:
                        polygons.append(poly)
            except Exception as error:
                print("error occured: ", error)
        print("generated polygons...")

        return polygons

    def segmentTissue(self, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False, mode='openslide'):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        if not os.path.exists(os.path.join(self.save_location, 'masks')):
            os.mkdir(os.path.join(self.save_location, 'masks'))

        if mode == 'openslide':
            wsi = openslide.OpenSlide(os.path.join(self.file_location, self.image_file))
            img = (wsi.read_region((0, 0), len(wsi.level_dimensions) - 1, wsi.level_dimensions[-1]).convert('RGB'))
            img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)  # Convert to HSV space
            img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring
        else:
            raise Exception("backend WSI reader not correctly specified...")

        vi = pyvips.Image.new_from_array(img)
        bands = utils.split_rgb(vi)
        colors = ["red", "green", "blue"]
        ink_color_perc = []
        for color in colors:
            perc = filters.pen_percent(bands, color)
            print(f"ink {color}: {perc:.5f}")
            ink_color_perc.append(perc)

        if ink_color_perc[0] > 0.08 or ink_color_perc[1] > 0.08 or ink_color_perc[2] > 0.08:
            raise Exception("ink detected on slide. aborting")


        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        if mode == 'openslide':
            scale = (wsi.level_downsamples[-1], wsi.level_downsamples[-1])
        else:
            xml_string = bioformats.get_omexml_metadata(os.path.join(self.file_location, self.image_file))
            o = bioformats.OMEXML(xml_string)
            x, y = o.image().Pixels.get_SizeX(), o.image().Pixels.get_SizeY()
            scale = (x / x_dim, y / y_dim)
            print(scale)

        save_data = Image.fromarray(img_otsu).convert('L')
        if '.svs' in self.save_name:
            save_data.save(os.path.join(self.save_location, 'masks', self.save_name.replace('.svs', '.png')))
        elif '.ndpi' in self.save_name:
            save_data.save(os.path.join(self.save_location, 'masks', self.save_name.replace('.ndpi', '.png')))
        elif '.czi' in self.save_name:
            save_data.save(os.path.join(self.save_location, 'masks', self.save_name.replace('.czi', '.png')))
        elif '.qptiff' in self.save_name:
            save_data.save(os.path.join(self.save_location, 'masks', self.save_name.replace('.qptiff', '.png')))
        else:
            pass
        return self.mask2polygons(img_otsu, scale)

    def measure_overlap(self, tile_starts, tile_ends, roi):
        ''' calculates overlap of tile with ROI regions '''
        tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [
            tile_ends[0], tile_ends[1]]
        tile_box = list(tile_box)
        tile_box = MultiPoint(tile_box).convex_hull
        ov = 0  # initialize
        if tile_box.intersects(roi):
            # box_label = True
            ov_reg = tile_box.intersection(roi)
            ov += ov_reg.area / tile_box.area
        return ov

    def save_hdf5(self, output_path, asset_dict, attr_dict=None, mode='w'):
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

    def extract_features_openslide(self, model, transforms, device, use_annotations=False, save_data = True):
        """
        Tesselate whole slide image into tiles and extract features for weakly supervised learning. Uses Bioformats in backend
        Uses bioformats image reader as backend for reading large WSIs
        :param use_annotations: if true, selects tiles from user-defined ROIs. Else selects tiles from entire tissue
        :param save_data: if true, saves extracted features to h5 format
        :return: {coords: [Nx2], features: [NxD]}. N: number of tiles analyzed. D: feature embedding dimension
        """
        model = model.to(device)
        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)

        if os.path.isfile(os.path.join(self.save_location,self.save_name + "_features.hdf5")):
            print("file already preprocessed... moving to next one")
            return
        #if not os.path.exists(os.path.join(self.save_location, self.save_name)):
        #    os.mkdir(os.path.join(self.save_location, self.save_name))

        ############ PARSE METADATA ############
        oslide = openslide.OpenSlide(os.path.join(self.file_location, self.image_file))

        try:
            # this is physical microns per pixel
            acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        except:
            try:
                acq_mag = 10.0 / (10000 / float(oslide.properties['tiff.XResolution']))
            except:
                print('MPP metadata missing. aborting')
                return



        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.patch_size * acq_mag / base_mag)

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(oslide, tile_size=physSize - round(self.pixel_overlap * acq_mag / base_mag),
                                  overlap=round(self.pixel_overlap * acq_mag / base_mag / 2),
                                  limit_bounds=self.limit_bounds)

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(
            base_mag / (tiles._l_z_downsamples[i] * tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in
            range(0, tiles.level_count))

        ############# GET ROIS ###############
        tumorshapes = list()
        if use_annotations:
            try:
                with open(self.xml_file) as f:
                    allobjects = geojson.load(f)

                allshapes = [shape(obj["geometry"]) for obj in allobjects]
                alllabels = [obj['properties'] for obj in allobjects]
                for roi_num in range(0, len(alllabels)):
                    try:
                        # I am assuming all the same class
                        tumorshapes.append(allshapes[roi_num])
                    except:
                        print('geometry loading problem')
                        pass
                # I only had 1 class, so I just did a unary_union on all tumorshapes
                tumorshapes = unary_union(
                    [geom.convex_hull if geom.geom_type == 'LineString' else geom for geom in tumorshapes])

            except:
                print('annotation file does not exist. aborting')
                return
        else:
            try:
                tumorshapes = unary_union(self.segmentTissue(4, close=3, use_otsu=True, mode="openslide"))
            except:
                print("Too much ink in slide " + self.image_file)
                return

        ########### SELECT & PROCESS TILES ##############
        wsi_coords = []
        wsi_features = []
        lvl = self.mag_extract
        if lvl in tile_lvls:
            x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]
            x_dim, y_dim = tiles.level_tiles[tile_lvls.index(lvl)]
            # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
            for y in tqdm(range(0, y_tiles)):
                batch = []
                coords_batch = []
                for x in range(0, x_tiles):
                    # grab tile coordinates
                    tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                    tile_ends = (
                    int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),
                    int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))

                    #measure overlap between tile and annotation
                    ov = self.measure_overlap(tile_starts=tile_coords[0], tile_ends=tile_ends, roi=tumorshapes)

                    # normally we only pull boxes that have a > 25% overlap
                    if ov > 0.25:
                        tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        tile_pull = tile_pull.resize(size=(self.patch_size, self.patch_size),
                                                     resample=Image.LANCZOS)
                        ws = self.whitespace_check(im=tile_pull)
                        if ws < 0.95:
                            try:
                                image = transforms(np.asarray(tile_pull))
                                image = image.unsqueeze(0)
                                batch.append(image)
                                coords_batch.append(tile_coords[0])
                            except:
                                print("error normalizing H&E stains... going to next image")
                                continue

                if len(batch) > 0:
                    #print(len(batch))
                    assert len(batch) == len(coords_batch)
                    tbatch = torch.cat(batch)
                    tbatch = tbatch.to(device)
                    model.eval()
                    with torch.no_grad():
                        features = model(tbatch)
                        features = features.cpu().numpy()
                        for i in range(features.shape[0]):
                            wsi_features.append(features[i,:])
                            wsi_coords.append(coords_batch[i])

            attr_dict = {'features': {'mag_level': 0,
                                      'wsi_name': self.save_name,
                                      'downsample': round(acq_mag / self.mag_extract),
                                      'level_dim': (x_dim, y_dim),
                                      'label': self.pt_label
                                      }}

            asset_dict = {'coords': np.vstack(wsi_coords), 'features': np.vstack(wsi_features)}

            if save_data:
                self.save_hdf5(os.path.join(self.save_location, self.save_name + '_features.hdf5'),
                               asset_dict=asset_dict, attr_dict=attr_dict)
            return asset_dict
        else:
            print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")
            return

    def extract_patches_openslide(self, use_annotations=False):
        """
        tesselate whole slide images into tiles
        """

        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)

        if not os.path.exists(os.path.join(self.save_location, self.save_name)):
            os.mkdir(os.path.join(self.save_location, self.save_name))

        ############ PARSE METADATA ############
        oslide = openslide.OpenSlide(os.path.join(self.file_location, self.image_file))

        try:
            # this is physical microns per pixel
            acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        except:
            try:
                acq_mag = 10.0 / (10000 / float(oslide.properties['tiff.XResolution']))
            except:
                print('MPP metadata missing. aborting')
                return



        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.patch_size * acq_mag / base_mag)

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(oslide, tile_size=physSize - round(self.pixel_overlap * acq_mag / base_mag),
                                  overlap=round(self.pixel_overlap * acq_mag / base_mag / 2),
                                  limit_bounds=self.limit_bounds)

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(
            base_mag / (tiles._l_z_downsamples[i] * tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in
            range(0, tiles.level_count))

        ############# GET ROIS ###############
        tumorshapes = list()
        if use_annotations:
            try:
                with open(self.xml_file) as f:
                    allobjects = geojson.load(f)

                allshapes = [shape(obj["geometry"]) for obj in allobjects]
                alllabels = [obj['properties'] for obj in allobjects]
                for roi_num in range(0, len(alllabels)):
                    try:
                        # I am assuming all the same class
                        tumorshapes.append(allshapes[roi_num])
                    except:
                        print('geometry loading problem')
                        pass
                # I only had 1 class, so I just did a unary_union on all tumorshapes
                tumorshapes = unary_union(
                    [geom.convex_hull if geom.geom_type == 'LineString' else geom for geom in tumorshapes])

            except:
                print('annotation file does not exist. aborting')
                return
        else:
            try:
                tumorshapes = unary_union(self.segmentTissue(4, close=3, use_otsu=True, mode="openslide"))
            except:
                print("Too much ink in slide " + self.image_file)
                return

        ########### SELECT & PROCESS TILES ##############
        lvl = self.mag_extract
        if lvl in tile_lvls:
            x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]
            x_dim, y_dim = tiles.level_tiles[tile_lvls.index(lvl)]
            # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
            for y in tqdm(range(0, y_tiles)):
                for x in range(0, x_tiles):
                    # grab tile coordinates
                    tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                    tile_ends = (
                        int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),
                        int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))

                    # measure overlap between tile and annotation
                    ov = self.measure_overlap(tile_starts=tile_coords[0], tile_ends=tile_ends, roi=tumorshapes)

                    # normally we only pull boxes that have a > 25% overlap
                    if ov > 0.25:
                        tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        ws = self.whitespace_check(im=tile_pull)
                        if ws < 0.95:
                            tile_pull = tile_pull.resize(size=(self.patch_size, self.patch_size),
                                                         resample=Image.LANCZOS)
                            tile_savename = '{}_{}_{}_{}.png'.format(tile_coords[0][0], tile_coords[0][1], physSize, self.patch_size)
                            tile_pull.save(os.path.join(self.save_location, self.save_name,
                                                        tile_savename))

        #with open(os.path.join(self.save_location, self.save_name, 'extraction_done.txt'), 'w') as f:
        #    pass
        #print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_location')
    parser.add_argument('--image_file')
    parser.add_argument('--mag', type=int, default=20, help='WSI magnification level')
    parser.add_argument('--patch_size', type=int, default=128, help='size of WSI image patches')
    parser.add_argument('--save_location', type = str, default = None, help='directory to save each patch')
    parser.add_argument('--save_name', type = str, default = None, help='name of slide directory')
    parser.add_argument('--xml_file', type = str, default = None, help='path to annotation file')
    parser.add_argument('--pt_label', type=str, default='unlabeled', help='pt level labels')
    args = parser.parse_args()

    ########### PREPARE PRETRAINED MODEL ##############
    local_dir = "/path/to/pre-trained/foundation/model/weights"
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "vit_large_patch16_224.dinov2.uni_mass100k.bin"), map_location="cpu"), strict=True)


    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrms_val = transforms.Compose(
        [
            NormalizeHEStains(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    ########### INITIALIZE WSI ANALYZER ###############
    wsi_analyzer = WSI_Analyzer(file_location=args.file_location, image_file=args.image_file,
                                save_location=args.save_location,
                                save_name=args.save_name, xml_file=args.xml_file, pt_label=args.pt_label,
                                mag_extract=args.mag, patch_size=args.patch_size)

    ############ RUN FEATURE EXTRACTION ################
    _ = wsi_analyzer.extract_features_openslide(model=model, transforms=trnsfrms_val, device=torch.device('cuda:0'))


