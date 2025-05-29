# Notes on preprocessing your whole slide H&E Image
1. For all datasets with SVS images we, use the [trident](https://github.com/mahmoodlab/trident) whole slide image preprocessing pipeline
2. For datasets with whole slide images from different scanners (*.czi, *.ndpi, *.mrxs), we utilize our own whole slide image tiling script. See [create_patches_custom.py](create_patches_custom.py). Following tiling we run foundation model feature extraction using the python script provided for each foundation model
3. For TCGA datasets, which were used for training, patches were extracted separately and we ran our own foundation model feature extraction scripts. These scripts incorporate Macenko stain normalization as an additional preprocessing step. Trident does not.
4. [create_patches_custom.py](create_patches_custom.py) uses [cuCim](https://github.com/rapidsai/cucim) in backend for fast preprocessing if the image is cuCim compatible. If not, uses standard openslide or bioformats for reading and tiling images (which is much slower!)
   
