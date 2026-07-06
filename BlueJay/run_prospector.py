"""
New modified model to work with new photometry on the cluster
Version that fits ONLY THE PHOTOMETRY!  NEW uses MIRI photometry
"""

import os
import numpy as np
import pandas as pd

from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import sedpy

from prospect.utils.obsutils import fix_obs
from prospect.sources import FastStepBasis
from prospect.models import transforms
from prospect.models.sedmodel import PolySpecModel
from prospect.models import priors
from prospect.models.templates import TemplateLibrary

import astropy.table as Table

def get_zred(galaxy_id):    
    # --- Read Blue Jay catalogue ---
    blue = "/Users/benjamincollins/University/Master/BlueJay/BlueJay_sample.txt"
    df = pd.read_csv(blue, sep='\s+')  # '\s+' handles whitespace-separated files

    # Faster lookup
    row = df.loc[df['id'] == int(galaxy_id)]
    z_spec = row['z_spec'].values[0]
    
    # Make sure that the code doesn't crash if it can't find the ID in the catalogue
    if len(row) == 0:   
        return None, False
    
    z_spec = row['z_spec'].values[0]
    
    if z_spec is not None and not np.isnan(z_spec):
        return z_spec, True
    else:
        z_phot = row['z_phot'].values[0]
        return z_phot, False

def flambda_to_maggies(wave_AA, flux):

    flux = flux * 1e-20 * u.erg/u.s/u.AA/u.cm**2
    fnu = flux * (wave_AA*u.AA)**2 / const.c
    fnu_Jy = fnu.to(u.Jy)
    fnu_maggies = fnu_Jy / 3631

    return fnu_maggies.value



def build_obs(objid, filt_list, mock_fit=False, fit_true_phot=False, **extras):
    """
    Modified to read the exact filter ordering from your BlueJay text file
    """
    # Read the filter paths directly from the text file
    # This automatically strips out newlines and keeps the exact order
    with open(filt_list, "r") as f:
        raw_filter_paths = [line.strip() for line in f if line.strip()]

    # Generate the proper Filter instances manually
    filter_names = []
    loaded_filters = []
    
    for path in raw_filter_paths:
        # Get filter name
        filter_name = os.path.basename(path)
        filter_names.append(filter_name)
        
        if 'alma_band6' in path:
            # For custom top-hat text files: load directly using sedpy.observate.Filter
            custom_filt = sedpy.observate.Filter(filename=path)
            custom_filt.name = filter_name # Give it your designated nick-name
            loaded_filters.append(custom_filt)
            
        elif 'alma_band7' in path:
            # For custom top-hat text files: load directly using sedpy.observate.Filter
            custom_filt = sedpy.observate.Filter(filename=path)
            custom_filt.name = filter_name # Give it your designated nick-name
            loaded_filters.append(custom_filt)
            
        else:
            # For native database filters: load via sedpy standard library
            # load_filters always returns a list, so we grab the first element [0]
            native_filt = sedpy.observate.load_filters([filter_name])[0]
            loaded_filters.append(native_filt)

    # Create the obs dictionary and load filters
    obs = {}    
    obs['filters'] = loaded_filters

    if mock_fit:
        # Load the mock CSV we created earlier
        path = f"/Users/benjamincollins/University/PhD/Code/bagpipes/BlueJay/data/mocks/{objid}_mock_phot.csv"
        df = pd.read_csv(path)
            
        # Reorder the DataFrame to match the filt_list order exactly
        df = df.set_index('filter_name').reindex(filter_names)
        
        # If user provided a specific list of bands, filter the data
        if fit_true_phot:
            # Use the true photometry (true_flux) for fitting
            flux_ujy = df['true_flux'].values
            print("⚠️ Using true photometry for fitting.")
        else:
            # Use mock photometry for fitting (default)
            flux_ujy = df['mock_flux'].values
            
        err_ujy = df['flux_err'].values
        
        obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])
    
    else:
        flux_ujy = []
        err_ujy = []
        phot_waves = []
        
        for path, loaded in zip(raw_filter_paths, loaded_filters):
            filter_name = os.path.basename(path)    # Leaves you with "acs_wfc_f606w"
            band = filter_name.split('_')[-1].upper()   # Leaves you with F606W
            
            # For HST and NIRCam look through the original Blue Jay table
            if 'hst' in path or 'nircam' in path:
                # Load HST/ACS and NIRCam photometry
                bluejay = fits.open("data/catalogues/bluejay_phot_cat_v1.4.fits")[1].data

                # Filter the catalogue to get the row corresponding to the galaxy ID
                row = bluejay[bluejay['id'] == int(objid)]
                
                # Access the flux and error for the specific band from the filtered row
                flux_jy = row[f'{band}_flux'][0]
                err_jy = row[f'{band}_flux_err'][0]
                
                # Convert Jy to µJy
                flux_ujy.append(flux_jy * 1e6)
                err_ujy.append(err_jy * 1e6)
                
                phot_waves.append(loaded.wave_effective)
            
            elif 'miri' in path:
                # Load MIRI photometry
                miri = fits.open("data/catalogues/Phot_Table_MIRI.fits")[1].data

                # Filter the catalogue to get the row corresponding to the galaxy ID
                row = miri[miri['id'] == int(objid)]
                
                # Access the flux and error for the specific band from the filtered row
                flux_jy = row[f'{band}_flux'][0]
                err_jy = row[f'{band}_flux_err'][0]
                
                # Convert Jy to µJy
                flux_ujy.append(flux_jy * 1e6)
                err_ujy.append(err_jy * 1e6)
                
                phot_waves.append(loaded.wave_effective)
            
            elif 'alma' in path:
                # Load ALMA data
                alma = fits.open("data/catalogues/ALMA_BlueJay.fits")[1].data

                # Filter the catalogue to get the row corresponding to the galaxy ID
                row = alma[alma['id'] == int(objid)]
                
                # Access the flux and error for the specific band from the filtered row
                flux_mjy = row['flux'][0]
                err_mjy = row['flux_err_sim'][0]
                
                # Convert mJy to µJy
                flux_ujy.append(flux_mjy * 1e3)
                err_ujy.append(err_mjy * 1e3)
                
                if 'band6' in path:
                    phot_waves.append(1286.73999 * 1e4)    # Effective wavelength of ALMA band 6
                elif 'band7' in path:
                    phot_waves.append(872.663024 * 1e4)    # Effective wavelength of ALMA band 7
            else:
                print(f"Warning: Unrecognized filter path '{path}'. Skipping this filter.")

        obs['phot_wave'] = np.array(phot_waves)
    
    for f, w in zip(obs['filters'], obs['phot_wave']):
        print(f"Filter: {f.name} | Effective Wavelength: {w/10000:.2f} µm")
    
    # Convert microjanskys to maggies
    obs['maggies'] = np.array(flux_ujy) * 1e-6 / 3631
    obs['maggies_unc'] = np.array(err_ujy) * 1e-6 / 3631
    
    # Enable all 17 photometry points for the fit
    obs['phot_mask'] = np.ones(len(loaded_filters), dtype='bool')
    
    # Set elements related to spectral fitting to None
    obs = fix_obs(obs)
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None
    
    return obs


# --------------------
# Set up model
# --------------------

# tie dust1 to dust2, with a prior centered on dust1=dust2
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2
    
# modify to increase nbins
nbins_sfh = 7
    
# Now exactly matches Bagpipes
def zred_to_agebins(zred, z_limit_sfh=20.0, nbins_sfh=7):
    tuniv = cosmo.age(zred).value*1e9   # Age of the universe at the observed redshift in years
    #tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9 # Maximum age bin edge corresponding to z_limit_sfh
    tbinmax = tuniv*0.95
    # Compute edges in logarithmic space
    log_edges = np.append(np.array([0.0, 6.7, 7.0]), np.linspace(7.0, np.log10(tbinmax), int(nbins_sfh-1))[1:])
    agelims = log_edges.tolist()
    # Format into Prospector's required (N, 2) shape array
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs


def build_model(objid, zred=None, waverange=None, add_duste=True,
                add_agn=False, add_neb = True, fit_afe=False,**extras):
    """Build a prospect.models.SedModel object

    :param zred: (optional, default: None)
        approximate value for the redshift, which is left as a free parameter.

    :param waverange: (optional, default: None)
        rest-frame wavelength range in angstrom; used to calculate polyorder.

    :returns model:
        An instance of prospect.models.SedModel
    """
    # continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    model_params = {}
    
    zred, has_spec_z = get_zred(objid)
    
    # If a spectroscopic redshift exists, fix it
    if has_spec_z:
        model_params['zred'] = {"N": 1, "isfree": False,
                                "init": zred,
                                "units": "redshift"}
    else:   # Otherwise, leave it as a free parameter with a Gaussian prior on the photometric value
        model_params['zred'] = {"N": 1, "isfree": True,
                                "init": zred,
                                "units": "redshift",
                                "prior": priors.Normal(mean=zred, sigma=0.05)}
        
    model_params['logzsol'] = {"N": 1, "isfree": True,
                               "init": -0.5,
                               "units": r"$\log (Z/Z_\odot)$",
                               "prior": priors.TopHat(mini=-2, maxi=0.50)}

    if fit_afe:
        model_params['afe'] = {"N": 1, "isfree": True,
                               "init": 0.0,
                               "units": r"$[\alpha/fe]$",
                               "prior": priors.TopHat(mini=-0.2, maxi=0.6)}
    else:
        model_params['afe'] = {"N": 1, "isfree": False,
                               "init": 0.0,
                               "units": r"$[\alpha/fe]$"}

    model_params["logt_wmb_hot"] = dict(N=1, isfree=False, init=10.0)

    # -------------------------
    
    model_params['f_outlier_phot'] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.00,
                                      "prior": priors.TopHat(mini=0, maxi=0.5)}
    
    model_params['nsigma_outlier_phot'] = {"N": 1,
                                          "isfree": False,
                                          "init": 50.0}
    
    
    # ----------------------------
    # --- Continuity SFH ----
    # ----------------------------
    # A non-parametric SFH model of mass in fixed time bins with a smoothness prior
        
    agebins = zred_to_agebins(zred=zred, nbins_sfh=nbins_sfh)
    nbins = agebins.shape[0]    

    print(f"SFH: Non-parametric SFH with {nbins} bins")
    
    # This is the *total*  mass formed, as a variable
    model_params["logmass"]    = {"N": 1, "isfree": True,
                                  "init": 10.,
                                  'units': "Solar masses formed",
                                  'prior': priors.TopHat(mini=7.5, maxi=13.)}

    # This will be the mass in each bin.  It depends on other free and fixed
    # parameters.  Its length needs to be modified based on the number of bins
    model_params["mass"]       = {'N': nbins, 'isfree': False,
                                  'init': (10**10.5)/nbins,
                                  'units': "Solar masses formed",
                                  'depends_on': transforms.logsfr_ratios_to_masses}

    # This gives the start and stop of each age bin.  It can be adjusted and its
    # length must match the length of "mass"
    model_params["agebins"]    = {'N': nbins, 'isfree': False,
                                  'init': agebins,
                                  'units': 'log(yr)'}
    
    # This controls the distribution of SFR(t) / SFR(t+dt). It has nbins-1 components.
    model_params["logsfr_ratios"] = {'N': nbins-1, 'isfree': True,
                                     'init': np.full(nbins-1, 0.0),  # constant SFH
                                     'units': '',
                                     'prior':priors.StudentT(mean=np.full(nbins-1, 0.0),
                                                             scale=np.full(nbins-1, 0.3), 
                                                             df=np.full(nbins-1, 2))}


    # ------------------------------
    # --- Initial Mass Function  ---
    # ------------------------------

    model_params['imf_type'] = {'N': 1, 'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': "FSPS index",
                             'prior': None}


    # ----------------------------
    # --- Dust Absorption ---
    # ----------------------------

    model_params['dust_type'] = {"N": 1, "isfree": False,
                                "init": 4,
                                "units": "FSPS index"}
                                
    model_params['dust2'] = {"N": 1, "isfree": True,
                             "init": 0.5,
                             "units": "optical depth at 5500AA",
                             "prior": priors.TopHat(mini=0.0, maxi=4.0/1.086)}

    model_params["dust_index"] = {"N": 1,
                                 "isfree": True,
                                 "init": 0.0, "units": "power-law multiplication of Calzetti",
                                 "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3)}

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}

    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}


    # ----------------------------
    # --- Dust Emission ---
    # ----------------------------

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_gamma']['init']  = 0.01
        model_params['duste_gamma']['prior'] = priors.TopHat(mini=0.0, maxi=1.0)
        
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['init']   = 3.5
        model_params['duste_qpah']['prior']  = priors.TopHat(mini=0.5, maxi=10.0)
        
        model_params['duste_umin']['isfree'] = True
        model_params['duste_umin']['init']   = 1.0
        model_params['duste_umin']['prior']  = priors.TopHat(mini=0.1, maxi=25.0)

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)


    if add_neb: 
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        model_params['nebemlineinspec'] = {'N': 1,
                                            'isfree': False,
                                            'init': False}
        _ = model_params["gas_logz"].pop("depends_on")

    # ----------------------------
    # ----------------------------
    # Now instantiate the model object using this dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------

def build_sps(zred, zcontinuous=1, smooth_instrument=False, obs=None, **extras):
    """
    :param zcontinuous:
        A value of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    sps = FastStepBasis(zcontinuous=zcontinuous)

    if (obs is not None) and (smooth_instrument):
        #from exspect.utils import get_lsf
        print('---- wave-dependent resolution ----')
        wave_obs = obs["wavelength"]
        sigma_v  = obs["sigma_v"]
        speclib  = sps.ssp.libraries[1].decode("utf-8")
        wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.set_lsf(wave, delta_v)

    return sps


def get_lsf(wave_obs, sigma_v, speclib, zred, **extras):
    """This method takes an instrimental resolution curve and returns the
    quadrature difference between the instrumental dispersion and the library
    dispersion, in km/s, as a function of restframe wavelength
    :param wave_obs: ndarray
        Observed frame wavelength (AA)
    :param sigma_v: ndarray
        Instrumental spectral resolution in terms of velocity dispersion (km/s)
    :param speclib: string
        The spectral library.  One of 'miles' or 'c3k_a', returned by
        `sps.ssp.libraries[1]`
    """
    lightspeed = 2.998e5  # km/s
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    wave_rest = wave_obs / (1 + zred)

    # Get the library velocity resolution function at the corresponding
    # *rest-frame* wavelength
    if speclib == "miles":
        miles_fwhm_aa = 2.54
        sigma_v_lib = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
        # Restrict to regions where MILES is used
        good = (wave_rest > 3525.0) & (wave_rest < 7500)

    elif speclib == "c3k_a":
        R_c3k = 3000
        sigma_v_lib = lightspeed / (R_c3k * 2.355)
        # Restrict to regions where C3K is used
        good = (wave_rest > 2750.0) & (wave_rest < 9100.0)

    elif speclib == "c3k_hr ":
        data_lib = np.loadtxt('/home/PERSONALE/letizia.bugiani2/fsps/SPECTRA/C3K/c3k_hr.lambda', 
                                dtype=[('wave_lib', '<f8'), ('sigma_v_lib', '<f8')])
        sigma_v_lib = data_lib['sigma_v_lib'][np.digitize(wave_rest, data_lib['wave_lib'])-1]
        good = (wave_rest > 0)

    else:
        sigma_v_lib = sigma_v
        good = slice(None)
        raise ValueError("speclib of type {} not supported".format(speclib))

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_lib**2, 0, np.inf))

    # return the broadening of the rest-frame library spectra required to match
    # the observed frame instrumental lsf
    return wave_rest[good], dsv[good]

# ------------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None




if __name__=='__main__':
    import time
    import sys
    import prospect.fitting
    import prospect.io.write_results
    
    # - Parser with default arguments -
    parser = prospect.utils.prospect_args.get_parser()

    # - Add custom arguments -
    parser.add_argument('--objid', type=int, default=0000,
                        help="ID of the object to fit")
    parser.add_argument('--output_tag', type=str, default="mock_test",
                        help="output tag name")          
    parser.add_argument('--add_duste', action="store_true", default=True,
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_neb', action="store_true",default=True,
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_agn', action="store_true", default=False,
                        help="If set, add agn emission to the model.")
    parser.add_argument('--fit_afe', action="store_true", default=False,
                        help="If set, use afe as a free parameter to fit.")
    parser.add_argument('--filt_list', type=str, default="filters/all_filters.txt",
                        help="Path to the filter list file.")
    parser.add_argument('--fit_true_phot', action="store_true", default=False,
                        help="If set, use true photometry for fitting.")
    

    args = parser.parse_args()
    run_params = vars(args)

    run_params["param_file"] = __file__
    run_params["outfile"] = ""
    
    # add in dynesty settings
    run_params['dynesty'] = True
    # Dynesty settings optimised for serial M1 MacBook performance
    run_params['dynesty'] = True
    run_params['nested_nlive_init'] = 1000          # Robust start, easier on RAM
    run_params['nested_nlive_batch'] = 500          # Good balance for batch updates
    run_params['nested_walks'] = 20                 # Reduced from 25 to lower CPU pressure
    run_params['nested_target_n_effective'] = 500   # Sufficient for high-quality posteriors
    run_params['nested_dlogz_init'] = 0.05          # Faster convergence; 0.01 is often overkill
    run_params['nested_maxcall'] = 2000000          # 5M is excessive for SEDs; 2M is plenty
    run_params['nested_maxcall_init'] = 2000000
    run_params['nested_sample'] = 'rwalk'           # Keep as rwalk for Prospector
    run_params['nested_maxbatch'] = None
    run_params['nested_first_update'] = {'min_ncall': 10000, 'min_eff': 7.5}


    print("\nFitting galaxy {}".format(run_params["objid"]))
    print("------------------\n")

    # build observations
    obs = build_obs(**run_params)
    #np.save('obs/obs_'+str(run_params["objid"]), obs)
    #print('obs saved')

    zred, is_spec = get_zred(run_params['objid'])
    run_params['zred'] = zred
    
    # build sps
    sps = build_sps(zred=run_params['zred'], smooth_instrument=False, obs=obs)
    
    # calculate rest-frame wavelength range 
    good_wave = obs['phot_wave'][obs['phot_mask']]
    run_params['waverange'] = (np.max(good_wave) - np.min(good_wave)) / (1 + run_params['zred'])

    # build model
    model = build_model(**run_params)

    # build noise
    noise = build_noise(**run_params)

    # Local execution (Skipping MPI setups to keep it simple on Mac)
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    print("Beginning Dynesty Sampling...")
    output = prospect.fitting.fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

    # Get unique name for the output file
    #hfile = "{0}_{1}_{2}_mcmc.h5".format(
    hfile = "{0}_{1}_mcmc.h5".format(
        run_params["objid"], 
        run_params["output_tag"]#,
        #int(time.time())
    )

    # Write results to file
    # Note: Make sure a directory called 'output/' exists in your working directory!
    out_dir = f'comparison/{run_params["output_tag"]}/output/'
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, hfile)
    
    prospect.io.write_results.write_hdf5(out_name, run_params, model, obs,
                                         output["sampling"][0], output["optimization"][0],
                                         tsample=output["sampling"][1],
                                         toptimize=output["optimization"][1])

    print(f"\nFinished! Results saved to {out_name}\n")