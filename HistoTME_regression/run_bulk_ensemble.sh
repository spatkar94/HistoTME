#!/bin/bash
SAVE_DIR="predictions"
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/UNI_v1_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed uni --save_loc $SAVE_DIR
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/UNI_v2_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed uni2 --save_loc $SAVE_DIR
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/Virchow_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed virchow --save_loc $SAVE_DIR
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/Virchow2_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed virchow2 --save_loc $SAVE_DIR
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/Gigapath_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed gigapath --save_loc $SAVE_DIR
python predict_bulk.py --h5_folder /mnt/synology/ICB_Data_SUNY/Hoptimus0_resection_256_embeddings --chkpts_dir logs --cohort SUNY_NSCLC --embed hoptimus0 --save_loc $SAVE_DIR
python gen_ensemble_predictions.py --mode bulk --cohort SUNY_NSCLC --save_loc $SAVE_DIR




