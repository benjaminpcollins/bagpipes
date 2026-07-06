#!/bin/bash

# Define the filter files and their respective IDs
# We use an associative array to store the IDs for each filter file
declare -A ALMA
ALMA["filters/bluejay_alma6.txt"]="8280"
ALMA["filters/bluejay_alma7.txt"]="13103 18252 21165"

# Loop through the dictionary
for filt_file in "${!ALMA[@]}"; do
    ids=${ALMA[$filt_file]}
    
    echo "--- Using Filter List: $filt_file ---"
    
    # Loop through the IDs for this filter list
    for objid in $ids; do
        echo "Processing Galaxy: $objid"
        
        # Run the python script
        python pipes_script.py --objid "$objid" --filt_file "$filt_file" --output_tag "bluejay_with_alma"
        
    done
done