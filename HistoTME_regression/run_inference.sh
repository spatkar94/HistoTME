#!/bin/bash
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/UNI_v1_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed uni
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/UNI_v2_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed uni2
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/Virchow_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed virchow
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/Virchow2_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed virchow2
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/Gigapath_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed gigapath
python predict_bulk.py --embeddings_folder /mnt/synology/ICB_Data_SUNY/Hoptimus0_resection_256_embeddings --cohort SUNY --cancer_type NSCLC --embed hoptimus0
python gen_ensemble_predictions.py --cohort SUNY --cancer_type NSCLC




