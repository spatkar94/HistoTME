#!/bin/bash
BASE_DIR="/your/base/dir"
FILE="your_h5py_file.h5"
SAVE_DIR="/home/air/Shared_Drives/MIP_network/MIP/spatkar/HistoTME/spatial_predictions"
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed uni --save_loc $SAVE_DIR
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed uni2 --save_loc $SAVE_DIR
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed virchow --save_loc $SAVE_DIR
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed virchow2 --save_loc $SAVE_DIR
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed gigapath --save_loc $SAVE_DIR
python predict_spatial.py --h5_path ${BASE_DIR}/${FILE} --chkpts_dir logs --embed hoptimus0 --save_loc $SAVE_DIR
python gen_ensemble_predictions.py --mode spatial --filename $FILE --save_loc $SAVE_DIR
