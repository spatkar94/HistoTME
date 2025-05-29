# Notes on preprocessing your WSI to extract foundation model feature embeddings
1. For all datasets with SVS images we, use the [trident](https://github.com/mahmoodlab/trident) whole slide image preprocessing pipeline
2. For datasets with whole slide images from different scanners (*.czi, *.ndpi, *.mrxs), we utilize our own whole slide image tiling script. See [create_patches.py](create_patches.py). Following tiling we run foundation model feature extraction using the python script provided for each foundation model
3. For TCGA datasets, which were used for training, patches were extracted separately and we ran our own foundation model feature extraction scripts. These scripts incorporate Macenko stain normalization as an additional preprocessing step. Trident does not.
   
