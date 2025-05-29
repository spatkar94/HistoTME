#!/bin/bash

ctype=("LUAD" "LSCC" "HNSCC" "UCEC" "CCRCC" "PDA" "GBM")
fm=("uni" "uni2" "virchow" "virchow2" "gigapath" "hoptimus0")
for item1 in "${ctype[@]}"; do
    for item2 in "${fm[@]}"; do
        python predict_CPTAC.py --task multitask --cohort CPTAC --cancer_type $item1 --embed $item2
    done 
done 