import os
import glob
import openslide
import argparse
from pathlib import Path
import cv2
import h5py
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import unary_union
import numpy as np
from cucim import CuImage
from cucim.clara.cache import preferred_memory_capacity
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import pandas as pd


class DeepZoomGeneratorCucim(DeepZoomGenerator):
    """Create a DeepZoomGenerator, but instead of utilizing OpenSlide,
    use cucim to read regions.

    Args:
        osr (OpenSlide): OpenSlide Image. Needed for OS compatibility and for retrieving metadata.
        cucim_slide (CuImage): CuImage slide. Used for retrieving image data.
        tile_size (int, optional): the width and height of a single tile.  For best viewer
                      performance, tile_size + 2 * overlap should be a power
                      of two.. Defaults to 254.
        overlap (int, optional): the number of extra pixels to add to each interior edge
                      of a tile. Defaults to 1.
        limit_bounds (bool, optional): True to render only the non-empty slide region. Defaults to False.
    """

    def __init__(
        self,
        osr: OpenSlide,
        cucim_slide: CuImage,
        tile_size: int = 256,
        overlap: int = 0,
        limit_bounds=False,
    ):
        super().__init__(osr, tile_size, overlap, limit_bounds)

        self._cucim_slide = cucim_slide
        self.memory_capacity = preferred_memory_capacity(
            self._cucim_slide, patch_size=(tile_size, tile_size)
        )
        self.cache = CuImage.cache(
            "per_process", memory_capacity=self.memory_capacity, record_stat=True
        )

    def get_tile(self, level: int, address: tuple[int]) -> Image:
        """Return an RGB PIL.Image for a tile

        Args:
            level (int): the Deep Zoom level
            address (tuple(int)): the address of the tile within the level as a (col, row)
                   tuple

        Returns:
            Image: PIL Image
        """
        args, z_size = self._get_tile_info(level, address)

        tile = self._cucim_slide.read_region(
            location=args[0],
            level=args[1],
            size=args[2],
        )
        tile = Image.fromarray(np.array(tile), mode="RGB")  # CuImage is RGB

        # Scale to the correct size
        if tile.size != z_size:
            # Image.Resampling added in Pillow 9.1.0
            # Image.LANCZOS removed in Pillow 10
            tile.thumbnail(z_size, getattr(Image, "Resampling", Image).LANCZOS)

        return tile


def stitching(file_path, wsi, downscale=64):
    bg_color = (0, 0, 0)
    alpha = -1
    draw_grid = False
    w, h = wsi.level_dimensions[0]
    print('original size: {} x {}'.format(w, h))

    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('downscaled size for stitching: {} x {}'.format(w, h))

    with h5py.File(file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        print('start stitching {}'.format(dset.attrs['name']))
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']

    print(f'number of patches: {len(coords)}')
    print(f'patch size: {patch_size} x {patch_size} patch level: {patch_level}')
    # 40x 0, 20x 1
    if (patch_level == 40 and patch_size == 256) or (patch_level == 20 and patch_size == 128):
        extract_level = 0
    elif patch_level == 20 and patch_size == 256:
        extract_level = 1
    else:
        extract_level = 1
        #import pdb;
        #pdb.set_trace()
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[extract_level]).astype(np.int32))
    print(f'ref patch size: {patch_size} x {patch_size}')

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        canvas = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        canvas = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    canvas = np.array(canvas)

    downsamples = wsi.level_downsamples[vis_level]
    indices = np.arange(len(coords))
    total = len(indices)

    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    for idx in tqdm(range(total)):
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            cv2.rectangle(canvas, tuple(np.maximum([0, 0], coord - 2 // 2)),
                          tuple(coord - 2 // 2 + np.array(patch_size)), (0, 0, 0, 255), thickness=2)
    return Image.fromarray(canvas)

class WSI_Analyzer:
    def __init__(self, by_folder, save_location, patch_size, mag_extract):
        self.save_location = save_location
        self.mag_extract = mag_extract  # specify which magnification you wish to pull images from - ONLY SUPPORTS SINGLE MAG
        self.patch_size = patch_size  # specify image size to be saved at specified magnification
        self.pixel_overlap = 0  # specify the level of pixel overlap in your saved images
        self.runlist = pd.read_csv(by_folder, header=None)
        print(self.runlist.shape)

        self.limit_bounds = True

    @staticmethod
    def whitespace_check(im):
        '''
        checks amount of whitespace in image
        :param im:
        :return: float: proportion of image covered in whitespace
        '''
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw = bw / 255
        prop_ws = (bw > 0.9).sum() / (bw > 0).sum()
        return prop_ws

    @staticmethod
    def mask2polygons(mask, scale):
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

    def segmentTissue(self, img, save_name):
        ''' create tissue mask '''

        # get he image and find tissue mask
        he = img.read_region((0, 0),  len(img.level_dimensions) - 1, img.level_dimensions[-1]).convert('RGB')
        # he = he[:, :, 0:3]
        heHSV = cv2.cvtColor(np.array(he), cv2.COLOR_RGB2HSV)
        he_blur = cv2.medianBlur(heHSV[:, :, 1], 7)  # Apply median blurring

        _, he_otsu = cv2.threshold(he_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # imagem = cv2.bitwise_not(he_otsu)
        # Morphological closing
        kernel = np.ones((3, 3), np.uint8)
        he_otsu = cv2.morphologyEx(he_otsu, cv2.MORPH_CLOSE, kernel)

        scale = (img.level_downsamples[-1], img.level_downsamples[-1])

        save_data = Image.fromarray(he_otsu).convert('L')
        if not os.path.exists(os.path.join(self.save_location, save_name, 'mask')):
            os.mkdir(os.path.join(self.save_location, save_name, 'mask'))
            
        save_data.save(os.path.join(self.save_location, save_name, 'mask',save_name+'.png'))

        return self.mask2polygons(he_otsu, scale)


    @staticmethod
    def measure_overlap(tile_starts, tile_ends, roi):
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

    @staticmethod
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
                try:
                    dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,dtype=data_type)
                except:
                    #import pdb; pdb.set_trace()
                    pass
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        print(key)
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
        file.close()
        print("finished writing to hdf5 file...")
        return output_path

    def create_patches_openslide(self, save_data = True, stitch = True):
        """
        Tesselate whole slide image into tiles and extract features for weakly supervised learning. Uses Bioformats in backend
        Uses bioformats image reader as backend for reading large WSIs
        :param stitch: generate downsampled stitched tiles to visually check the results.
        :param use_annotations: if true, selects tiles from user-defined ROIs. Else selects tiles from entire tissue
        :param save_data: if true, saves extracted features to h5 format
        :return: {coords: [Nx2], features: [NxD]}. N: number of tiles analyzed. D: feature embedding dimension
        """
        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)

        #flist = sorted(glob.glob(self.runlist + '/*.svs'))
        flist = list(self.runlist.iloc[:,0].values)
        coords = []
        ############ PARSE METADATA ############
        for _file in flist:
            print(_file)
            oslide = OpenSlide(_file)
            savnm = os.path.basename(_file)
            save_name = str(Path(savnm).with_suffix(''))
            if not os.path.exists(os.path.join(self.save_location, save_name)):
                os.mkdir(os.path.join(self.save_location, save_name))
            if not os.path.isfile(os.path.join(self.save_location, save_name, save_name + ".hdf5")):
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
                try:
                    oslide_cu = CuImage(_file)
                    tiles = DeepZoomGeneratorCucim(osr=oslide, cucim_slide=oslide_cu,
                                                   tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag),
                                                   overlap=round(self.pixel_overlap*acq_mag/base_mag/2),
                                                   limit_bounds=self.limit_bounds)
                    print("try using cucim first...")
                except Exception as e:
                    tiles = DeepZoomGenerator(osr=oslide,
                                              tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag),
                                              overlap=round(self.pixel_overlap*acq_mag/base_mag/2),
                                              limit_bounds=self.limit_bounds)
                    print("cucim not compatible, using openslide...")


                # calculate the effective magnification at each level of tiles, determined from base magnification
                tile_lvls = tuple(
                    base_mag / (tiles._l_z_downsamples[i] * tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for
                    i in range(0, tiles.level_count))

                ########### SELECT & PROCESS TILES ##############
                wsi_coords = []
                lvl = self.mag_extract
                if lvl in tile_lvls:
                    # send to get tissue polygons
                    try:
                        print('detecting tissue...')
                        tissue = unary_union(self.segmentTissue(oslide, save_name))
                    except:
                        print('tissue not found or too much ink on slide')
                        continue
                    x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]
                    print('creating tiles...')
                    for y in tqdm(range(0, y_tiles)):
                        for x in range(0, x_tiles):
                            tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                            tile_ends = (
                                int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),
                                int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))
                            # measure overlap between tile and annotation
                            ov = self.measure_overlap(tile_starts=tile_coords[0], tile_ends=tile_ends, roi=tissue)
                            # normally we only pull boxes that have a > 25% overlap
                            if ov > 0.9:
                                tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                                tile_size = tiles.get_tile_dimensions(tile_lvls.index(lvl), (x, y))
                                if tile_size != (physSize, physSize):
                                    tile_pull = Image.fromarray(cv2.copyMakeBorder(np.array(tile_pull), 0,
                                                                                   physSize - int(tile_size[1]),
                                                                                   0,physSize - int(tile_size[0]),
                                                                                   cv2.BORDER_REFLECT))
                                    

                                ws = self.whitespace_check(im=tile_pull)
                                if ws < 0.8:
                                    try:
                                        wsi_coords.append(tile_coords[0])
                                        #print(tile_coords[1])
                                        save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1])
                                        tile_savename = save_name + "_" + str(self.mag_extract) + "_"  + save_coords + "_" + "ws-" + '%.2f' % (ws) + "_" + "ov-" + '%.2f' % (ov)
                                        if not os.path.exists(os.path.join(self.save_location, save_name,'patches')):
                                            os.mkdir(os.path.join(self.save_location, save_name,'patches'))
                                        tile_pull.save(os.path.join(self.save_location, save_name,'patches',tile_savename + ".png"))
                                    except Exception as error:
                                        print("error normalizing H&E stains... going to next image")
                                        print("error msg: ", error)
                                        continue
                else:
                    print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")
                    continue

                if len(wsi_coords) >0:
                    asset_dict = {'coords': np.vstack(wsi_coords)}
                    attr = {'patch_size': self.patch_size,
                            'patch_level': lvl,
                            'downsample': round(acq_mag / self.mag_extract),
                            'level_dim': (x_tiles, y_tiles),
                            'name': save_name,
                            'save_path': self.save_location}

                    attr_dict = {'coords': attr}

                    if save_data:
                        self.save_hdf5(os.path.join(self.save_location, save_name, save_name + '.hdf5'),
                                       asset_dict=asset_dict, attr_dict=attr_dict)

                    if stitch:
                        file_path = os.path.join(self.save_location, save_name,save_name + '.hdf5')
                        if os.path.isfile(file_path):
                            try:
                                heatmap = stitching(file_path, oslide, downscale=64)
                                stitch_path = os.path.join(self.save_location, save_name, save_name + '.jpg')
                                heatmap.save(stitch_path)
                            except Exception as e:
                                print(e)


            else:
                print("file already preprocessed... moving to next one")
                patchify_path = os.path.join(self.save_location,save_name, save_name + ".hdf5")

                with h5py.File(patchify_path, "r") as file:
                    print('coordinates size: ', file['coords'].shape)
                    coords.append(file['coords'].shape[0])
        print(len(coords))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # change to BCMC or UW folder
    parser.add_argument('--by_folder', default="/data/MIP/harmon-lab/blca/WSI/Fred_Hutch/slides_all", type=str)
    parser.add_argument('--mag', type=int, default=20, help='WSI magnification level')
    parser.add_argument('--patch_size', type=int, default=128, help='size of WSI image patches')
    parser.add_argument('--save_location', type = str, default = "/data/MIP/harmon-lab/blca/WSI/Fred_Hutch/patch_40_256/patches", help='directory to save each patch')
    # add model path for cancer detection
    argus = parser.parse_args()

    ########### INITIALIZE WSI ANALYZER ###############
    wsi_analyzer = WSI_Analyzer(by_folder=argus.by_folder, save_location=argus.save_location, mag_extract=argus.mag, patch_size=argus.patch_size)

    ############ RUN FEATURE EXTRACTION ################
    wsi_analyzer.create_patches_openslide()


