#!/bin/bash
filename='TCGA_LUSC.txt'
echo Start
while read p; do
  python embed_wsi_UNI.py \
  --file_location /path/to/svs/files \
  --image_file $p \
  --mag 20 \
  --patch_size 512 \
  --save_location /path/to/save/embeddings \
  --save_name $p;
done < "$filename"
