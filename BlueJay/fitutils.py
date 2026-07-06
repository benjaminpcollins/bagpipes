import os
import glob
import numpy as np
import pickle as pkl
import pandas as pd
import prospect.io.read_results as reader
from prospect.models.transforms import logsfr_ratios_to_sfrs
from prospect.utils.plotting import get_percentiles, get_best
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

def load_prospector_results(galaxy_id, prosp_dir):
    """Function to load Prospector result .h5 files and disect its 
    data structure to be used for easy plotting.

    Args:
        galaxy_id (int): The ID of the galaxy for which to load results
        prosp_dir (str): The directory containing the Prospector output files

    Returns:
        dict: A comprehensive dictionary storing the parameters samples, MAP values and quantiles
    """
    
    # Load the h5 file for the given galaxy ID
    h5_files = glob.glob(os.path.join(prosp_dir, f'*{galaxy_id}*.h5'))
    try:
        h5_file = h5_files[0]
        print(f"Loading Prospector output file: {h5_file}")
    except IndexError:
        print(f"No PROSPECTOR results found for objid {galaxy_id}.")
        return None

    # Load PROSPECTOR results
    results, obs, model = reader.results_from(h5_file)
        
    # Now we have to exclude the last 3 parameters from the fit
    map_parameters = get_best(results)
    
    # Extract labels for parameters that were "free" (fitted)
    labels = map_parameters[0]

    # Build the MAP dictionary
    MAP = {}
    for a,b in zip(map_parameters[0], map_parameters[1]):
        MAP[a] = b
    
    # Extract chains, weights and the MAP index
    chain = results['chain']
    weights = results['weights']
    imax = np.argmax(results['lnprobability'])
    
    data = {
            'meta': {'labels': map_parameters[0], 'map_idx': imax, 'weights': weights},
            'params': {}
        }

    perc = get_percentiles(results, [16, 50, 84])    
    
    for i, name in enumerate(map_parameters[0]):
            # Use the flattened chain for statistics
            param_samples = chain[:, i]
            
            if name == 'dust2':         # convert optical depth to mag
                data['params'][name] = {
                'samples': param_samples * 1.086,
                'map': MAP[name] * 1.086,   # Use the value from the best vector directly
                'q16': perc[name][0] * 1.086,
                'q50': perc[name][1] * 1.086,
                'q84': perc[name][2] * 1.086
            }
                
            else:
                data['params'][name] = {
                    'samples': param_samples,
                    'map': MAP[name],   # Use the value from the best vector directly
                    'q16': perc[name][0],
                    'q50': perc[name][1],
                    'q84': perc[name][2]
                }
    return data


def load_bagpipes_results(galaxy_id, bagp_dir):
    """Function to load Bagpipes result .h5 files and disect its 
    data structure to be used for easy plotting.

    Args:
        galaxy_id (int): The ID of the galaxy for which to load results
        bagp_dir (str): The directory containing the Bagpipes output files

    Returns:
        dict: A comprehensive dictionary storing the parameters samples, MAP values and quantiles
    """
    
    file = os.path.join(bagp_dir, f'{galaxy_id}.h5')    # Only one file per galaxy
    
    print(f"Loading Bagpipes output file: {file}")
    
    with h5py.File(file, 'r') as results:
        
        # Get redshift of the source
        fit_str = results.attrs['fit_instructions']
        fit = eval(fit_str, {"np": np, "array": np.array})
        zred = fit['redshift']
        
        # Extract the sampling results
        chain = results['samples2d']
        
        # Get maximum likelihood inde
        imax = np.argmax(results['lnlike'])
        
        # Manually extracted the labels from the results
        labels = ['dsfr1', 'dsfr2', 'dsfr3', 'dsfr4', 'dsfr5', 'dsfr6', 'logmass', 'logzsol', 'dust2', 
                    'duste_gamma', 'dust_index', 'duste_qpah', 'duste_umin', 'gas_logu']
        
        # Build the data structure
        data = {
            'meta': {'labels': labels, 'map_idx': imax, 'weights': None},
            'params': {}
        }
        
        # Loop through each parameter
        for i, name in enumerate(labels):
            samples = chain[:, i]
            q16, q50, q84 = quantile(samples, [0.16, 0.5, 0.84], weights=None)
            
            if name == 'logzsol':   # Convert metallicity to log
                data['params'][name] = {
                    'samples': np.log10(samples),
                    'map': np.log10(samples[imax]),
                    'q16': np.log10(q16),
                    'q50': np.log10(q50),
                    'q84': np.log10(q84)
                }
            
            else:    
                data['params'][name] = {
                    'samples': samples,
                    'map': samples[imax],
                    'q16': q16,
                    'q50': q50,
                    'q84': q84
                }

        data['params']['zred'] = {'samples': None, 'map': zred, 'q16': zred, 'q50': zred, 'q84': zred}
        
    return data


def plot_dual_corner(galaxy_id, data1, label1, data2, label2, title=f"Galaxy", mock_data=None, 
                     save_dir="/Users/benjamincollins/University/PhD/Code/bagpipes/BlueJay/comparison/mock_fit", 
                     scale_dust1=False, scale_dust2=True,
                     colour1="orange", colour2="dodgerblue",
                     save_fig=True):
    """Figure to plot corner plot of two posterior distributions, given that they are stored in the same format.

    Args:
        galaxy_id (int) 
            ID of the galaxy
        data1 (dict): 
            Output data of the first fit
        label1 (str): 
            String describing the first fit
        data2 (dict): 
            Output data of the second fit
        label2 (str): 
            String describing the second fit
        mock_data (dict): 
            Dictionary containing the real galaxy properties in case of mock fit, otherwise None
        save_dir (str): 
            Output directory for the figure
        scale_dust1 (bool):
            Rescale dust2 parameter from optical thickness to Av (Only needed for Prospector)
        scale_dust2 (bool):
            Rescale dust2 parameter from optical thickness to Av (Only needed for Prospector)
        colour1 (str):
            Colour for first dataset (default: orange)
        colour2 (str):
            Colour for second dataset (default: dodgerblue)
    
    """
    # Define internal keys and display labels
    plot_keys = ['logmass', 'logzsol', 'dust2', 'duste_gamma', 'dust_index', 'duste_qpah', 'duste_umin', 'gas_logu']
    
    labels = [r"$\log_{10}(M_*)$", r"$\log_{10}(Z/Z_\odot)$", r"$A_V$", 
              r"Dust $\gamma$", "Dust Index", r"Dust $q_{PAH}$", 
              r"Dust $U_{min}$", r"$\log_{10}(U)$"]

    # Extract and Transform Samples
    a_samps = np.array([data1['params'][k]['samples'] for k in plot_keys]).T
    b_samps = np.array([data2['params'][k]['samples'] for k in plot_keys]).T
    
    a_weights = data1['meta'].get('weights') if 'meta' in data1 else None
    b_weights = data2['meta'].get('weights') if 'meta' in data2 else None
    
    a_weights = np.ones(len(a_samps)) if a_weights is None else a_weights
    b_weights = np.ones(len(b_samps)) if b_weights is None else b_weights
    
    samples_list = [a_samps, b_samps]
    sample_labels = [label1, label2]
    colors = [colour1, colour2]

    # Calculate Global Range (So both fits are visible)
    ndim = b_samps.shape[1]
    plot_range = []
    for dim in range(ndim):
        dim_min = min(np.nanmin(a_samps[:, dim]), np.nanmin(b_samps[:, dim]))
        dim_max = max(np.nanmax(a_samps[:, dim]), np.nanmax(b_samps[:, dim]))
        plot_range.append([dim_min, dim_max])

    # Base Corner Settings
    shared_kwargs = dict(
        labels=labels,
        range=plot_range,
        smooth=0.9,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=False, # We disable automatic titles to avoid overlaps
        max_n_ticks=3,
        hist_kwargs=dict(density=True)
    )

    # First
    fig = corner.corner(
        a_samps,
        labels=labels,
        range=plot_range,
        color=colors[0],
        label_kwargs={"fontsize": 22},
        weights=a_weights,
        smooth=0.9,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=False,
        max_n_ticks=4,
        hist_kwargs={'color': colors[0], 'linewidth': 2, 'density': True}
    )

    # Second
    # We turn off 'fill_contours' for the second one so we can see through it
    corner.corner(
        b_samps,
        fig=fig,
        range=plot_range,
        color=colors[1],
        label_kwargs={"fontsize": 22}, 
        weights=b_weights,
        smooth=0.9,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True, 
        show_titles=False,
        max_n_ticks=4,
        hist_kwargs={'color': colors[1], 'linewidth': 2, 'density': True}
    )

    # 7. Add Legend and Title
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i], lw=4)
            for i in range(len(colors))
        ],
        fontsize=28, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    
    # Only add text box with true parameters if we have mock data!
    if mock_data:
        true_params = mock_data["true_params"][()]

        real_values = "Mock values:\n\n"
        for i, p in enumerate(true_params.values()): # iterating mock_data dictionary
                real_values += f"{labels[i]}: {p}\n"
        
        box_style = dict(
            boxstyle='round,pad=0.5', # Shapes: 'square', 'round', 'larrow', etc.
            facecolor='wheat',         # Inside color of the box
            edgecolor='orange',        # Border color
            alpha=0.5                  # Transparency (0 to 1)
        )
        
        plt.text(-5.6, 4.0, s=real_values, fontsize=28, bbox=box_style)
    
    ndim = a_samps.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    
    for i in range(ndim):
        ax = axes[i, i]
        
        # 2. Extract stats for Bagpipes (Orange)
        a_p = data1['params'][plot_keys[i]]
        a_val, a_plus, a_minus = a_p['q50'], a_p['q84'] - a_p['q50'], a_p['q50'] - a_p['q16']
        
        # 3. Extract stats for Prospector (Black/Blue)
        b_p = data2['params'][plot_keys[i]]
        b_val, b_plus, b_minus = b_p['q50'], b_p['q84'] - b_p['q50'], b_p['q50'] - b_p['q16']
        
        # Special Case: If it's Av (index 2), apply the 1.086 scale to the text labels too
        if i == 2:
            if scale_dust1:
                a_val, a_plus, a_minus = a_val*1.086, a_plus*1.086, a_minus*1.086
            if scale_dust2:
                b_val, b_plus, b_minus = b_val*1.086, b_plus*1.086, b_minus*1.086
        
        # 4. Create the strings
        # Use \text{} or raw strings to handle the LaTeX formatting
        a_str = f"${a_val:.2f}^{{+{a_plus:.2f}}}_{{-{a_minus:.2f}}}$"
        b_str = f"${b_val:.2f}^{{+{b_plus:.2f}}}_{{-{b_minus:.2f}}}$"

        # 5. Set the title
        # We use a newline \n to stack them. Note: 'y' controls the vertical height.
        # 4. Place individual text objects (Manually colored)
        # x=0.5 centers it. y=1.02 and 1.15 stack them above the plot.
        ax.text(0.5, 1.23, a_str, color="orange", transform=ax.transAxes, 
                fontsize=22, ha='center', va='bottom', fontweight='bold')
        
        ax.text(0.5, 1.03, b_str, color="dodgerblue", transform=ax.transAxes, 
                fontsize=22, ha='center', va='bottom', fontweight='bold')
        
        # 6. Coloring the text
        # To get specific colors for specific lines of the title, 
        # we can use ax.annotate or just rely on the labels in the legend.
        # But for absolute clarity, we can color the whole title block:
        #ax.title.set_color('black') # Or 'darkgrey' to be neutral
    
    
    
    fig.suptitle(title, fontsize=32, y=1.0, fontweight="bold")
    
    if save_fig:
        fig_path = os.path.join(save_dir, f'{galaxy_id}_corner.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    plt.close()
    
    
    
    
def plot_sfh_prospector(h5_file, figname=None):
    # Load PROSPECTOR results
    results, obs, model = reader.results_from(h5_file)

    imax = np.argmax(results['lnprobability'])

    map_parameters = results['chain'][imax, :].copy()

    # Build the MAP dictionary
    MAP = {}
    for a,b in zip(results['theta_labels'], map_parameters):
        MAP[a] = b

    zred = MAP['zred']
    logmass = MAP['logmass']
    agebins = results['model_params'][8]['init']    # 8 for agebins

    print(len(agebins))

    dt = 10**agebins[:, 1] - 10**agebins[:, 0]

    # Collect logsfr_ratios
    logsfr_ratios = np.array([MAP[f"logsfr_ratios_{i}"] for i in range(1, len([k for k in MAP if k.startswith("logsfr_ratios_")])+1)])        
    # Convert to SFRs
    sfh_best = logsfr_ratios_to_sfrs(logmass, logsfr_ratios, agebins)

    # Sample from the chains!
    n_steps = results['chain'].shape[0]
    # Take 500 weighted posterior samples from the chain
    sample_indices = np.random.choice(n_steps, size=500, p=results['weights']/np.sum(results['weights']))
    # Get the full set of parameters from the chain
    samples = results['chain'][sample_indices, :]

    sfh_samples = []
    for params_i in samples:
        new_map = {}
        for a,b in zip(results['theta_labels'], params_i):
            new_map[a] = b
        # Get logsfr_samples
        logsfr_sample = np.array([new_map[f"logsfr_ratios_{i}"] for i in range(1, len([k for k in new_map if k.startswith("logsfr_ratios_")])+1)])
        sfh_sample = logsfr_ratios_to_sfrs(logmass, logsfr_sample, agebins)
        sfh_samples.append(sfh_sample)

    # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
    sfh_lower = np.percentile(sfh_samples, 16, axis=0)
    sfh_median = np.percentile(sfh_samples, 50, axis=0)
    sfh_upper = np.percentile(sfh_samples, 84, axis=0)

    # Convert log age bins to linear time (yr)
    bin_edges = 10**agebins  # shape (nbins, 2)

    bin_edges *= 1e-9

    bin_starts = bin_edges[:, 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the Median SFH
    ax.step(bin_starts, sfh_median, where='post', color='black', lw=2, label='Median SFH')

    # Plot the MAP SFH
    ax.step(bin_starts, sfh_best, where='post', color='crimson', lw=2, label='MAP SFH')

    # Plot the Uncertainty (16th-84th percentile)
    ax.fill_between(bin_starts, sfh_lower, sfh_upper, step="post", color='gray', alpha=0.4)

    # Formatting
    ax.set_xlabel('Lookback Time (Gyr)', fontsize=14)
    ax.set_ylabel('SFR ($M_\odot yr^{-1}$)', fontsize=14)
    #ax.invert_xaxis() # Crucial: Lookback time goes from now (left) to past (right)
    ax.legend()
    #plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    if figname:
        plt.savefig(f"/Users/benjamincollins/University/PhD/Code/bagpipes/BlueJay/pipes/plots/mock_fit/{figname}")
    plt.show()
    
    
    
def find_modes(samples, weights=None, prominence=0.05, mass_threshold=0.2):
    """
    Returns peaks that contain at least 'mass_threshold' (e.g., 20%) of posterior mass.
    """
    if weights is None: weights = np.ones(len(samples))
    
    # 1. Find peaks using KDE
    kde = gaussian_kde(samples, weights=weights)
    x_grid = np.linspace(samples.min(), samples.max(), 500)
    density = kde.evaluate(x_grid)
    peak_indices, _ = find_peaks(density, prominence=prominence * density.max())
    peak_values = x_grid[peak_indices]
    peak_densities = density[peak_indices]
    
    if len(peak_values) == 0: return []
    
    # 2. Assign each sample to its closest peak
    # (Reshaping for broadcasting: samples [N,1] - peaks [1, P])
    dist = np.abs(samples[:, np.newaxis] - peak_values[np.newaxis, :])
    closest_peak_idx = np.argmin(dist, axis=1)
    
    # 3. Calculate mass fraction for each peak
    total_weight = np.sum(weights)
    significant_data = []
    
    for i in range(len(peak_values)):
        peak_mass = np.sum(weights[closest_peak_idx == i]) / total_weight
        
        if peak_mass >= mass_threshold:
            # We store the value AND the density so we can sort later
            significant_data.append((peak_values[i], peak_densities[i]))
            
    if not significant_data: return np.array([])
    
    # 4. Sort by density (index 1) in descending order
    significant_data.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the values, now guaranteed to be density-sorted
    return np.array([item[0] for item in significant_data])

def check_for_overlap(peaks1, peaks2, tolerance=0.1):
    """
    Checks if any peak in code 1 is 'near' a peak in code 2.
    tolerance: how close peaks need to be (in normalized parameter space)
    """
    overlaps = []
    for p1 in peaks1:
        for p2 in peaks2:
            if abs(p1 - p2) < tolerance:
                overlaps.append((p1, p2))
    return overlaps





def plot_corner(galaxy_id, data, color="Orange", title="Galaxy"):
    # Select the labels you want to see
    # Using your existing labels: ['logmass', 'logzsol', 'dust2', 'gas_logu']
    plot_labels = ['logmass', 'logzsol', 'dust2', 'duste_gamma', 'dust_index', 'duste_qpah', 'duste_umin', 'gas_logu']
    
    # Extract the samples for these specific labels
    samples = np.array([data['params'][l]['samples'] for l in plot_labels]).T
    
    # Create the corner plot
    fig = corner.corner(
        samples,
        labels=[r"$\log_{10}(M_*)$", r"$\log_{10}(Z/Z_\odot)$", r"$A_V$", r"Dust $\gamma$", "Dust Index", r"Dust $q_{PAH}$", r"Dust $U_{min}$", r"Dust $U_{min}$", r"$\log_{10}(U)$"],
        quantiles=[0.16, 0.5, 0.84],
        weights=data['meta']['weights'],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        color=color,
        smooth=1.0, # Helps visualize bimodality with only 500 samples
        # --- ADD THESE THREE LINES ---
        plot_datapoints=False,  # Suppresses the black dots
        fill_contours=True,     # Fills the 1/2/3 sigma levels with color
        plot_density=False      # Suppresses the grey background 'cloud'
    )
    
    fig.suptitle(title, fontsize=16)
    plt.show()
