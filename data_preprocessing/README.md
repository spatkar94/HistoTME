# Notes on preprocessing your whole slide H&E Image
1. For all datasets with SVS images, the [trident](https://github.com/mahmoodlab/trident) whole slide image preprocessing pipeline can be utilized.
2. For datasets with whole slide images from different scanners (ndpi, mrxs), we utilize our own whole slide image tiling script. See [create_patches_custom.py](create_patches_custom.py). Following tiling we run foundation model feature extraction using the python scripts provided for each foundation model. Note: You will need to set up a access token for each foundation model on huggingface prior to running these scripts
3. For WSI datasets generated using Zeiss digital slide scanner (czi), trident provides scripts to convert the images into openslide compatible format (*.tif). Once converted you can then run trident on the converted images to obtain foundation model features.
4. For TCGA datasets, which were used for pan-cancer training, patches were extracted using our own custom scripts (at the time, Trident wasn't available). These scripts incorporate Macenko stain normalization as an additional preprocessing step.
5. [create_patches_custom.py](create_patches_custom.py) uses [cuCim](https://github.com/rapidsai/cucim) in the backend for fast WSI preprocessing if the image is cuCim compatible. If not, uses standard openslide pipelines (which are significantly slower!)
   
