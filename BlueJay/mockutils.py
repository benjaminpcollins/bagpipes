import os
import glob
import numpy as np
import pickle as pkl
import pandas as pd
from corner import quantile
import h5py

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"  # override bagpipes' Helvetica request
mpl.rcParams["text.usetex"] = True

os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/Library/TeX/texbin'

# Increase tick label size and thickness
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.linewidth'] = 1.5  # Makes the box frames thicker

from astropy.cosmology import WMAP9 as cosmo
#%matplotlib inline

from astropy.io import fits
from astropy.table import Table
from sedpy import observate

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Redshift
def get_zred(galaxy_id):    
    # --- Read Blue Jay catalogue ---
    blue = "/Users/benjamincollins/University/Master/BlueJay/BlueJay_sample.txt"
    tbl = Table.read(blue, format="ascii.basic")
    
    row = tbl[tbl['id'] == int(galaxy_id)]
    
    # Make sure that the code doesn't crash if it can't find the ID in the catalogue
    if len(row) == 0:   
        return None, False
    
    z_spec = row['z_spec'][0]
    
    if z_spec is not None and not np.isnan(z_spec):
        return z_spec, True
    else:
        z_phot = row['z_phot'][0]
        return z_phot, False

# Star formation histories
def zred_to_agebins(zred, z_limit_sfh=20.0, nbins_sfh=8):
    tuniv = cosmo.age(zred).value*1e9   # Age of the universe at the observed redshift in years
    #tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9 # Maximum age bin edge corresponding to z_limit_sfh
    tbinmax = tuniv*0.95
    # Compute edges in logarithmic space
    log_edges = np.append(np.array([0.0, 6.7, 7.0]), np.linspace(7.0, np.log10(tbinmax), int(nbins_sfh-1))[1:])
    bin_edges = 10**log_edges   # Convert back to linear space
    bin_edges /= 1e6 # ensure that edges are in Myr for Bagpipes
    return bin_edges.tolist()   # return list of age bin edges

def write_alma_transmission_curves(file_path, central_freq_ghz, bandwidth_ghz):
    c_m_per_s = 299792458   # Speed of light in m/s
    
    # Calculate frequency edges (Hz)
    nu_min = (central_freq_ghz - (bandwidth_ghz / 2.0)) * 1e9
    nu_max = (central_freq_ghz + (bandwidth_ghz / 2.0)) * 1e9
    
    # Convert to Wavelength edges (Angstroms)
    wave_start = (c_m_per_s / nu_max) * 1e10
    wave_end = (c_m_per_s / nu_min) * 1e10
    
    # Create a 4-point perfect rectangle
    # We add a tiny buffer (e.g., 0.1 Angstrom) to make the filter walls perfectly vertical
    wavelengths = np.array([
        wave_start - 0.1,  # Just outside the left wall
        wave_start,        # Just inside the left wall
        wave_end,          # Just inside the right wall
        wave_end + 0.1     # Just outside the right wall
    ])
    
    # Assign transmission values (We do 1.0 since ALMA bands are very narrow)
    transmissions = np.array([0.0, 1.0, 1.0, 0.0])
    
    # Generate a two-column text file for Bagpipes
    output_data = np.column_stack((wavelengths, transmissions))
    
    filename = file_path + ".par"
    
    np.savetxt(
        filename, 
        output_data, 
        fmt=['%.4f', '%.1f'], 
        comments=''
    )
    print(f"💾 Saved tophat filter to: {filename}")
    

def extract_true_params(model_components):
    # 1. Identify which SFH component is active (the one that isn't nebular, redshift, or dust)
    sfh_types = ["constant", "exponential", "lognormal", "dblplaw"]
    active_sfh_key = next((key for key in model_components if key in sfh_types), None)
    
    # 2. Build the parameter dict
    params = {
        "redshift": model_components["redshift"],
        "sfh_type": active_sfh_key,
    }
    
    # 3. Dynamically add all SFH parameters (mass, metallicity, tau, etc.)
    if active_sfh_key:
        sfh_params = model_components[active_sfh_key]
        for key, val in sfh_params.items():
            # Log-transform metallicity if it's found
            if key == 'metallicity':
                params['logzsol'] = np.log10(val)
            elif key == 'massformed':
                params['logmass'] = val
            else:
                params[f"sfh_{key}"] = val
    
    # 4. Dynamically add all Dust parameters
    for key, val in model_components['dust'].items():
        params[f"dust_{key}"] = val
        
    # 5. Dynamically add all Nebular parameters
    for key, val in model_components['nebular'].items():
        params[f"gas_{key}"] = val
        
    return params


def make_loader(filtlist, fit_true_phot=False):
    """
    A 'factory' that creates a custom loading function 
    with the selected filters baked into it.
    """
    def loader(objid):
        # Your load_data logic here
        return load_mock_data(objid, filtlist=filtlist, fit_true_phot=fit_true_phot)
    
    return loader

def load_mock_data(objid, filtlist, fit_true_phot=False):    
    """
    Load photometry from a mock CSV or real FITS catalogues.
    
    objid: The ID of the galaxy
    data_source: 'mock' or 'real'
    desired_filters: List of strings (e.g., ['jwst_f444w', 'alma_band6']) 
                     If None, it loads all available.
    """

    # Extract the filter names from the filtlist
    filter_names = [os.path.basename(f) for f in filtlist]
    
    # Load the mock CSV we created earlier
    path = f"/Users/benjamincollins/University/PhD/Code/bagpipes/BlueJay/data/mocks/{objid}_mock_phot.csv"
    df = pd.read_csv(path)
        
    # Reorder the DataFrame to match the filt_list order exactly
    df = df.set_index('filter_name').reindex(filter_names)
    
    # If user provided a specific list of bands, filter the data
    if fit_true_phot:
        # Use the true photometry (true_flux) for fitting
        fluxes = df['true_flux'].values
        print("⚠️ Using true photometry for fitting.")
    else:
        # Use mock photometry for fitting (default)
        fluxes = df['mock_flux'].values
        
    errs = df['flux_err'].values

    # Enforce SNR limits / missing data handling
    photometry = np.c_[fluxes, errs]
    for i in range(len(photometry)):
        if (photometry[i, 0] <= 0.) or (np.isnan(photometry[i, 0])):
            photometry[i, :] = [0., 9.9e99]
            
    return photometry



def load_bluejay_with_alma(ID):
    """ 
    Load BlueJay photometry from the BlueJay catalogue(s)
        
        Note: Pay attention to which filtlist you are using 
        since this determines which ALMA band is specified!
    
    """

    # Blue Jay catalogue
    bluejay_cat = Table.read("data/catalogues/bluejay_phot_cat_v1.4.fits")
    
    # 1. List all HST/ACS and NIRCam bands:
    filters = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W', 'F606W', 'F814W']
    
    # 2. Find the correct row using the ID column
    # Use a mask rather than (int(ID) - 1) to be safe against non-sequential IDs
    row = bluejay_cat[bluejay_cat['ID'] == int(ID)]

    if len(row) == 0:
        raise ValueError(f"ID {ID} not found in catalogue.")
    
    # 3. Extract fluxes and errors into lists
    fluxes = []
    flux_errs = []

    for f in filters:
        fluxes.append(row[f + "_flux"][0] * 1e6)
        flux_errs.append(row[f + "_flux_err"][0] * 1e6)


    # MIRI catalogue
    miri_cat = Table.read("data/catalogues/Phot_Table_MIRI.fits")
    
    # 1. List all available MIRI bands:
    miri_filters = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    
    # 2. Find the correct row using the ID column
    # Use a mask rather than (int(ID) - 1) to be safe against non-sequential IDs
    row = miri_cat[miri_cat['ID'] == int(ID)]

    if len(row) == 0:
        raise ValueError(f"ID {ID} not found in catalogue.")
    
    # 3. Extract fluxes and errors into lists
    for f in miri_filters:
        fluxes.append(row[f + "_flux"][0] * 1e6)
        flux_errs.append(row[f + "_flux_err"][0] * 1e6)
        
    # ALMA catalogue
    alma_cat = Table.read("data/catalogues/ALMA_BlueJay.fits")
    
    # 1. Load the ALMA data and check the wavelengths
    row = alma_cat[alma_cat['ID'] == int(ID)]
    
    # 2. Convert fluxes from mJy to microJy (1 mJy = 1000 microJy)
    fluxes.append(row['flux'][0] * 1e3) 
    flux_errs.append(row['flux_err_sim'][0] * 1e3)
    
    # Now turn these into a 2D array [N_filters, 2]
    # Bagpipes expects photometry[i, 0] = flux, photometry[i, 1] = error
    photometry = np.c_[fluxes, flux_errs]
    
    # 5. Clean up missing data and enforce SNR limits
    for i in range(len(photometry)):
        # Blow up errors for missing data (NaN or 0 flux)
        if (photometry[i, 0] <= 0.) or (np.isnan(photometry[i, 0])):
            photometry[i, :] = [0., 9.9e99]
            continue # Skip SNR check for bad data
    
    return photometry