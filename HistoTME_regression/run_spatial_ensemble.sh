#!/bin/bash
FILE="UR-PDL1-LR-093-E-4-31-H&E"
SAVE_DIR="/home/air/Shared_Drives/MIP_network/MIP/spatkar/HistoTME/spatial_predictions"
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/UNI_v1_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed uni --save_loc $SAVE_DIR
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/UNI_v2_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed uni2 --save_loc $SAVE_DIR
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/Virchow_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed virchow --save_loc $SAVE_DIR
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/Virchow2_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed virchow2 --save_loc $SAVE_DIR
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/Gigapath_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed gigapath --save_loc $SAVE_DIR
python predict_spatial.py --h5_path /mnt/synology/ICB_Data_SUNY/Hoptimus0_resection_256_embeddings/${FILE}_features.hdf5 --chkpts_dir logs --embed hoptimus0 --save_loc $SAVE_DIR
python gen_ensemble_predictions.py --mode spatial --filename $FILE --save_loc $SAVE_DIR