#!/bin/bash

# List of dataset names
names=('IFNG' 'MHCI' 'MHCII' 'Coactivation_molecules' 'Effector_cells' 'T_cell_traffic' 'NK_cells' 'T_cells' 'B_cells' 'M1_signatures' 'Th1_signature' 'Antitumor_cytokines' 'Checkpoint_inhibition' 'Treg' 'T_reg_traffic' 'Neutrophil_signature' 'Granulocyte_traffic' 'MDSC' 'MDSC_traffic' 'Macrophages' 'Macrophage_DC_traffic' 'Th2_signature' 'Protumor_cytokines' 'CAF' 'Matrix' 'Matrix_remodeling' 'Angiogenesis' 'Endothelium' 'Proliferation_rate' 'EMT_signature')


# Loop through each name
for index in "${!names[@]}"; do
    name="${names[$index]}"
    # Print the index and the current name
    echo "Processing index $index: $name"

    save_arg="abmil_${name}_huber"
    # Run the python command with dataset name replaced
    python run.py --save="$save_arg" --dataset="$name" --bag_size=-1 --epochs=40
    
done
