#!/bin/bash
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_uni_v1 --cohort Gulley --embed uni
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_uni_v2 --cohort Gulley --embed uni2
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_virchow --cohort Gulley --embed virchow
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_virchow2 --cohort Gulley --embed virchow2
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_gigapath --cohort Gulley --embed gigapath
python predict_dataset.py --embeddings_folder /mnt/synology/ICB_Data_Gulley/gulley_embeddings/20x_256px_0px_overlap/features_hoptimus0 --cohort Gulley --embed hoptimus0
python gen_ensemble_predictions --cohort Gulley




