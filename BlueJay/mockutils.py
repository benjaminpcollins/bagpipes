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
    
    np.savetxt(
        file_path, 
        output_data, 
        fmt=['%.4f', '%.1f'], 
        comments=''
    )
    print(f"💾 Saved tophat filter to: {file_path}")
    

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