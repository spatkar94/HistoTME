HistoTME
==============
![](HistoTME_regression/HistoTME_outline.png)


## Overview 
HistoTME is a weakly supervised deep learning framework designed to infer cell type and pathway specific transcriptomic signature activity directly from whole slide H&E images, utilizing digital pathology foundation model feature embeddings. The code in the folder HistoTME_regression can be used to train HistoTME to predict TME signature activity from histopathology slides. The code in the folder HistoTME_downstream provides examples of how HistoTME signature predictions can be utilized for unsupervised clustering analyses and stratification of cancer patients responses to immunotherapy. The original HistoTME paper is available [here](https://www.nature.com/articles/s41698-024-00765-w).

## Introducing HistoTMEv2:
HistoTMEv2 has been trained and tested on 25 different cancer types. See our new preprint [here]()

## Installation and prerequisites
Tested with Python 3.9. Install requirements using:
```
pip install -r requirements.txt
```
Note: the preprocessing script makes use of NVIDIAs cuCIM image processing library. To install cuCIM, see instructions [here](https://github.com/rapidsai/cucim).

## How to use
### Data Preparation
#### Whole slide imaging data
The TCGA and CPTAC whole slide imaging and tanscriptomic data can be found online from [GDC](https://portal.gdc.cancer.gov/), [TCIA](https://wiki.cancerimagingarchive.net/display/Public/CPTAC+Imaging+Proteomics) data portals. The downloaded whole slide images should be stored in a single directory as shown below:
```bash
├── WSI_Directory
│   ├── slide_1.svs
│   ├── slide_2.svs
│   ├── ...
...
...
│   ├── slide_N.svs

```
After downloading the WSI, utilize the scripts provided in the [data_preprocessing](data_preprocessing) folder to tesselate each WSI into non-overlapping tiles and extract foundation model embeddings. 
Note: Our latest model tesselates WSI at 256x256 pixel, 20x magnification, in order to facilitate head-to-head benchmarking against other spatial transcriptomic prediction methods.
The extracted features will be saved in a h5py file with each entry corresponding to a tile along with its physical coordinates and the foundation model-generated feature embeddings.
```
dict{'coords': (x,y), 'features': <embeddings>}
```
#### Transcriptomics signatures data
To calculate ground truth activity of TME-associated signatures from bulk transcriptomics data please see the [following github repository](https://github.com/BostonGene/MFP/blob/master/TME_Classification.ipynb). 

### Format Preparation
The extracted features should be in h5py file format to be read. The ground truth transcriptomic signatures should be saved in a csv file format. See [example_data](example_data).
## Training
we provided updated scripts for training HistoTMEv2 in a pan-cancer 5-fold cross-validation fashion:
```
cd HistoTME_regression/
./run_training.sh
```

## Inference
We have provided updated scripts for running inference using HistoTMEv2, our pan cancer model. HistoTMEv2 can be run in two modes: bulk and spatial. Bulk mode generates signature scores for the whole slide image or patient. Whereas spatial mode generates tile-level scores

For bulk mode run the following. This code operates on the entire cohort
```
cd HistoTME_regression/
python predict_bulk.py [-h] [--h5_folder H5_FOLDER] [--cohort COHORT] [--cancer_type CANCER_TYPE]
                       [--num_workers NUM_WORKERS] [--embed EMBED]
```

For spatial mode run the following. This code operates on a single H&E image
```
cd HistoTME_regression/
python predict_spatial.py [-h] [--h5_path H5_PATH] [--num_workers NUM_WORKERS] [--embed EMBED]
                          [--save_loc SAVE_LOC] 
```

## Model weights
For inquiries about HistoTME model weights please contact the corresponding authors directly.  The codes are intended to be used for research purposes only. Please see the [license](LICENSE)

## Questions and Issues
If you find any bugs or have any questions about this code please contact: [Sushant Patkar](patkar.sushant@nih.gov) or [Alex Chen](alche@sas.upenn.edu)

## Citation
If you found our work useful in your research please consider citing this work as follows: 
```
@article{patkar2024predicting,
  title={Predicting the tumor microenvironment composition and immunotherapy response in non-small cell lung cancer from digital histopathology images},
  author={Patkar, Sushant and Chen, Alex and Basnet, Alina and Bixby, Amber and Rajendran, Rahul and Chernet, Rachel and Faso, Susan and Kumar, Prashanth Ashok and Desai, Devashish and El-Zammar, Ola and others},
  journal={NPJ Precision Oncology},
  volume={8},
  number={1},
  pages={280},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgments
This project was supported by an award from Upstate Foundation's Hendricks Endowment. Data (digital images and clinical meta-data) from the institutional cohort was generated at the Pathology Research Core Lab using institutional resources and support. 

