HistoTME
==============
![](figures/HistoTME_outline.png)


## Overview 
HistoTME is a weakly supervised deep learning model that infers cell type– and pathway-specific transcriptomic signature activity directly from H&E-stained whole-slide images (WSIs), without requiring single-cell resolution or explicit patch-level annotations. The model leverages patch embeddings generated from large-scale digital pathology foundation models and a multiple instance learning-based regression module to predict bulk transcriptomic signature activity from whole slide H&E images. This design enables HistoTME to learn biologically grounded representations of the tumor microenvironment (TME), bridging image-based and transcriptomic modalities.

The `HistoTME_regression` module contains code to train models for predicting the activity of 29 curated gene expression signatures from histopathology slides. The `HistoTME_downstream` module demonstrates use cases of these predictions for unsupervised TME profiling and stratification of immunotherapy response, illustrating how transcriptomic signature inference from WSIs can support scalable, annotation-free biomarker discovery. The original HistoTME paper is available [here](https://www.nature.com/articles/s41698-024-00765-w).

## Introducing HistoTMEv2 :rocket:
HistoTMEv2 is a pan-cancer extension of HistoTME, trained and tested on 25 different cancer types. See our new preprint [here](https://www.biorxiv.org/content/10.1101/2025.06.11.658673v1)


![](figures/pancancer_figure.png)

## Installation and prerequisites
First clone the repo and cd into the directory:
```
git clone https://github.com/spatkar94/HistoTME.git
cd HistoTME
```
Then create a conda env and install the dependencies:
```
conda create -n histoTME python=3.10 -y
conda activate histoTME
pip install -e .
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
After downloading the WSI, utilize the scripts provided in the [data_preprocessing](data_preprocessing) folder to tesselate each WSI into non-overlapping tiles and extract foundation model embeddings. Alternatively, you can also use [trident](https://github.com/mahmoodlab/trident) to preprocess your WSI (if they are svs images). The embeddings for each WSI are saved as a h5 file along with the coordinates of each tile:
```
dict{'coords': (x,y), 'features': <embeddings>}
```
Note: Our latest model, HistoTMEv2, tesselates each WSI into tiles of size 256x256 pixels, captured at 20x magnification, in order to facilitate head-to-head benchmarking against other spatial transcriptomic prediction methods. 

#### Transcriptomics data
To calculate ground truth activity of TME-associated signatures from bulk transcriptomics data please see the [following github repository](https://github.com/BostonGene/MFP/blob/master/TME_Classification.ipynb). The ground truth transcriptomic signatures should be saved in a csv file format. See [example_data](example_data).

### Training
we provided updated scripts for training HistoTME in a pan-cancer 5-fold cross-validation fashion:
```
cd HistoTME_regression/
./run_training.sh
```

### Inference
We have provided updated scripts for running inference. Our latest model can now be run in two modes: bulk and spatial. Bulk mode generates enrichment scores for the whole slide or patient. Whereas spatial mode generates tile-level enrichment scores.

![](figures/inference_modes_figure.png)

After generating tile embeddings with a foundation model you can run the following script to generate bulk signature predictions for a cohort of WSIs
```
cd HistoTME_regression/
python predict_bulk.py [-h] [--h5_folder H5_FOLDER] [--chkpts_dir CHKPTS_DIR] [--cohort COHORT]
                       [--save_loc SAVE_LOC] [--num_workers NUM_WORKERS] [--embed EMBED]

options:
  -h, --help            show this help message and exit
  --h5_folder H5_FOLDER
                        Path to directory containing h5py files
  --chkpts_dir CHKPTS_DIR
                        path to directory where pretrained model checkpoints are saved
  --cohort COHORT       Cohort name
  --save_loc SAVE_LOC   path to where predictions should be saved
  --num_workers NUM_WORKERS
  --embed EMBED         name of foundation model used: [uni, uni2, virchow, virchow2, gigapath, hoptimus0]
```

Alternatively you can also generate spatially resoloved predictions of signatures for a single WSI using the following python script
```
cd HistoTME_regression/
python predict_spatial.py [-h] [--h5_path H5_PATH] [--chkpts_dir CHKPTS_DIR] [--num_workers NUM_WORKERS]
                          [--embed EMBED] [--save_loc SAVE_LOC]

options:
  -h, --help            show this help message and exit
  --h5_path H5_PATH     WSI patch embeddings file path for prediction (single h5py file)
  --chkpts_dir CHKPTS_DIR
                        path to directory where pretrained model checkpoints are saved
  --num_workers NUM_WORKERS
  --embed EMBED         name of foundation model used: [uni, uni2, virchow, virchow2, gigapath, hoptimus0]
  --save_loc SAVE_LOC

```
### Combining signature predictions generated from multiple foundation models
Our cross-validation results indicate that combining signature predictions generated from several foundation models yeilds the most robust results. To generate ensemble predictions, run the following scripts below. Note: Remember to update the paths to where each foundation model's embeddings are stored prior to running the bash scripts.

For bulk predictions:
```
cd HistoTME_regression/
./run_bulk_ensemble.sh
```

For spatial predictions:
```
cd HistoTME_regression/
./run_spatial_ensemble.sh
```


## Model weights
Pretrained model checkpoints are available on [huggingface](https://huggingface.co/spatkar94/HistoTMEv2). After gaining access to the model here, you will need to login to HuggingFace in the environment you wish to use the model. You can do this from python as follows:

```
from huggingface_hub import login
AUTH_TOKEN = '' #specify your authentication token
login(AUTH_TOKEN)
```

For more details on logging in and downloading models from huggingface, please read their [documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). After logging in, you can download the weights to a specified path using the following python script
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id='spatkar94/HistoTMEv2', local_dir='local_dir')
```

This will save all pretrained model checkpoints to a folder named `checkpoints` under your specified local_dir. Specify this full path (i.e `local_dir/checkpoints`) as an argument to the `--chkpts_dir' option for running inference as shown above. 



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

