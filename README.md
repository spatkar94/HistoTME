HistoTME
==============

# Predicting tumor microenvironment composition and immunotherapy response in non-small cell lung cancer from digital histopathology images 

## Overview 
Implementation of HistoTME in our paper "Predicting tumor microenvironment composition and immunotherapy response in non-small cell lung cancer from digital histopathology images":
[Citation]

The code in the folder HistoTME_regression can be used to run attention-based multiple instance learning (ABMIL) to predict TME signatures derived from bulk transcriptomics. In order to run experiments on the histopathology datasets, please download the features extracted by the UNI foundation model for the TCGA and CPTAC H&E whole slide images (WSI).

The code in the folder HistoTME_downstream can be used to run downstream prediction of immune checkpoint inhibitor response in NSCLC patients. Prior to running these experiments, features must be extracted from the WSIs. _______ stuff on clustering downstream

## Installation and prerequisites
Tested with Python 3.8. Tested on both GPU (should I test on CPU?). Install requirements using:
```
pip install -r requirements.txt
```
## How to use
### Data Preparation
#### TCGA and CPTAC data
The TCGA and CPTAC data can be found in the following links: [][]. The data should be preprocessed using:
```
PREPROCESS SCRIPT
```

Deconvolution of bulk-transcriptomics into TME signatures can be calculated using:
```
Deconvolution script
```

#### Format Preparation
The extracted features should be in h5py file format to be read. A csv containing both TCGA and CPTAC cohorts should then be made with the transcriptomic-derived TME signatures and a file path to the extracted features. See HistoTME_regression/sample_TCGA_CPTAC.csv for an example. 

### Training
Training can be run for multi-task ABMIL or single-task ABMIL using:
```
HistoTME_regression/run_multitask.sh
HistoTME_regression/run_single_tasks.sh
```

### Prediction
Predictions can be run on the SUNY cohort using:
```
python HistoTME_regression/predict_SUNY.py --task=multitask
python HistoTME_regression/predict_SUNY.py --task=singletask
```
Predictions using multitask or singletask ABMIL can be run on the CPTAC or TCGA cohort using:
```
python HistoTME_regression/predict_CPTAC_TCGA.py --task=multitask --cohort=CPTAC
python HistoTME_regression/predict_CPTAC_TCGA.py --task=singletask --cohort=CPTAC
python HistoTME_regression/predict_CPTAC_TCGA.py --task=multitask --cohort=TCGA
python HistoTME_regression/predict_CPTAC_TCGA.py --task=singletask --cohort=TCGA
```

## Questions and Issues
If you find any bugs or have any questions about this code please contact Sushant or Alex.

## Citation
If you found our work useful in your research please consider citing our paper:

## Acknowledgments



Testing
