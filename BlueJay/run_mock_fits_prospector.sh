#!/opt/homebrew/bin/bash

filt_list="filters/hst_wfc_nircam_w_miri_all.txt"

#objid=9996

# Run the python script
#python run_prospector.py --objid "$objid" --filt_list "$filt_list" --output_tag "hst_wfc_nircam_w_miri_all" --mock_fit --zred 1.00

#python run_prospector.py --objid "$objid" --filt_list "$filt_list" --output_tag "hst_wfc_nircam_w_miri_all_TRUE_PHOT" --fit_true_phot --mock_fit --zred 1.00


objid=9995

# Run the python script
python run_prospector.py --objid "$objid" --filt_list "$filt_list" --output_tag "hst_wfc_nircam_w_miri_all" --fit_mock --zred 2.50

python run_prospector.py --objid "$objid" --filt_list "$filt_list" --output_tag "hst_wfc_nircam_w_miri_all_TRUE_PHOT" --fit_true_phot --fit_mock --zred 2.50
