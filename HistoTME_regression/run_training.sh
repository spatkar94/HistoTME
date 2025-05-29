#!/bin/bash

# List of task names
tasks=('antitumor' 'protumor' 'angio' 'cancer')
# foundation model name
fm='virchow'
# embeddings folder
path='/mnt/synology/Virchow_Pan_TCGA'
# GPU device
device='cuda:2'
# Loop through tasks
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"
    # Print the index and the current task
    echo "Processing index $index: $task"

    save_arg="abmil_${task}_huber"
    # Run the python command with task name replaced
    python run.py \
        --save="$save_arg" \
        --dataset="$task" \
        --bag_size=-1 \
        --epochs=40 \
        --embed="$fm" \
        --embeddings_folder="$path" \
        --device="$device"
    
done
