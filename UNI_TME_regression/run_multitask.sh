#!/bin/bash

# List of dataset names
names=('antitumor' 'protumor' 'angio' 'cancer')

# Loop through each name
for index in "${!names[@]}"; do
    name="${names[$index]}"
    # Print the index and the current name
    echo "Processing index $index: $name"

    save_arg="abmil_${name}_huber"
    # Run the python command with dataset name replaced
    python run.py --save="$save_arg" --dataset="$name" --bag_size=-1 --epochs=40
    
done
