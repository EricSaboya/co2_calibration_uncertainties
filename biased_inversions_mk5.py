# ------------------------------------------------------------------------------------------------- #
# Uncertainties in CO2 Calibration Reference Materials: Inverse Modelling Experiments
# ------------------------------------------------------------------------------------------------- #
# Created: 10 July 2024
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# ------------------------------------------------------------------------------------------------- #
# Functions for performing biased inversions of CO2 under a MAP scheme 
# ------------------------------------------------------------------------------------------------- #
import os
import sys
import copy 
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from collections import namedtuple

sys.path.append("/user/work/wz22079/projects/CO2/inversions/")
import utils
import convert
import inversion_setup as setup
import calculate_basis_functions as cbf
from get_co2_data import get_forward_sims

np.random.seed(141095)

def inputs():
    """
    Function defining input dictionaries used for the inversions 
    Edit as needed 
    """
    # Dictionary containing observations data specifications
    obs_dict = {"species": "co2",
                "site": ["CBW", "MHD", "BSD", "HFD", "RGL", "TAC", "WAO"],
                "inlet": ["207m", "24m", "248m", "100m", "90m", "185m", "10m"],
                "averaging_period": ["4H", "4H", "4H", "4H", "4H", "4H", "4H"],
                "instrument": ["multiple", "multiple", "picarro", "picarro", "picarro", "picarro", "multiple"],
                "data_level": ["2", "2", None, None, None, None, "2"],
                "store": "paris_obs_store",
                "calibration_scale": None,
                "start_date": "2021-05-01",
                "end_date": "2021-06-01",
                "filters": ["daytime"],
               }
    
    # Dictionary containing CO2 fluxes data specifications 
    flux_dict = {"species": "co2", 
                 "domain": "EUROPE",
                 "source": ["paris-bio-base", "paris-fossil-base"], 
                 "start_date": "2021-05-01",
                 "end_date": "2021-06-01",
                 "store": "co2_europe",
                }

    # Footprints dictionary CO2 data specifications
    fp_dict = {"species": "co2",
               "domain": "EUROPE",
               "site" : ["CBW", "MHD", "BSD", "HFD", "RGL", "TAC", "WAO"],
               "fp_height": ["200m", "10m", "248m", "100m", "90m", "185m", "20m"],
               "start_date": "2021-05-01",
               "end_date": "2021-06-01",
               "store": "co2_footprints",            
              }

    # Boundary conditions dictionary specifications
    bc_dict = {"species": "co2",
               "domain": "EUROPE",
               "bc_input" : "camsv22",
               "start_date": "2021-05-01",
               "end_date": "2021-06-01",
               "store": "co2_europe",
               "bc_freq": None,
              }

    # Basis functions dictionary 
    basis_dict = {"fp_basis_case": None,
                  "basis_directory": None, 
                  "fp_basis_algorithm": "weighted",
                  "nbasis": [80, 80],
                  "bc_basis_case": "NESW",
                  "bc_basis_directory": "/group/acrg/LPDM/bc_basis_functions/",
                 }

    # MCMC dict
    mcmc_inputs_dict = {"xprior": {"paris-bio-base": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
                                   "paris-fossil-base": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.3, "lower":0.0},
                                  },
                        "bcprior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1},
                        "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 3.0}, 
                        "add_offset": False, 
                        "offsetprior": None, 
                        "nit": 10000,
                        "burn": 1000, 
                        "tune": 2000,
                        "nchain": 4,
                        "sigma_per_site": True
                       }
                        
    return obs_dict, flux_dict, fp_dict, bc_dict, basis_dict, mcmc_inputs_dict

################################################################################################################

def get_co2_data(use_bc: bool,
                 outputname: str,
                 outputpath: str,
                ):
    """
    Function that retrieves necessary CO2 datasets 
    for mole fraction inversions 
    
    NB. This function does not add biases to the 
    pseudo-observations

    -----------------------------------------------
    Args:
        use_bc (bool):
            Option to run inversions with boundary conditions data
        outputname (str):
            Name used for saving any output files
        outputpath (str):
            Path to save output file

    Returns
        fp_data (dict):
            Dictionary containing forward simulations for each site 
        sites (list/str):
            List of sitenames that have data available during period 
            of interest
    """
    
    # Get dataset inputs for RHIME CO2 inversions 
    obs_dict, flux_dict, fp_dict, bc_dict, basis_dict, mcmc_inputs_dict = inputs()

    # Get forward simulations data for CO2
    data_dict, sites, inlet, fp_height, instrument, averaging_period = get_forward_sims(flux_dict, 
                                                                                        fp_dict, 
                                                                                        obs_dict, 
                                                                                        bc_dict, 
                                                                                        use_bc,
                                                                                       )
    if sites is None:
        sites = obs_dict["site"]
        inlet = obs_dict["inlet"]
        fp_height = fp_dict["fp_height"]
        instrument = obs_dict["instrument"]
        averaging_period = obs_dict["averaging_period"]

    basis_dict["site"] = sites 
    basis_dict["source"] = flux_dict["source"]
    basis_dict["domain"] = fp_dict["domain"]
    basis_dict["start_date"] = obs_dict["start_date"]
       
    # Calculate basis functions for each flux sector
    fp_data, tempdir, basis_dir, bc_basis_dir = cbf.basis_functions_wrapper(data_dict,
                                                                            basis_dict,
                                                                            use_bc=use_bc,
                                                                            outputname=outputname,
                                                                            outputpath=outputpath,
                                                                           )

    # Apply data filtering approaches
    if obs_dict["filters"] is not None:
        fp_data = utils.filtering(fp_data, obs_dict["filters"])

    # Remove any sites that return empty data array post-filtering
    s_dropped = []
    for site in sites:
        if fp_data[site].time.values.shape[0] == 0:
            s_dropped.append(site)
            del fp_data[site]
            
    if len(s_dropped) != 0:
        sites = [s for i, s in enumerate(sites) if s not in s_dropped]
        print(f"\nDropping {s_dropped} sites as no data passed the filtering.\n")

    # Append model domain region to site attributes
    for i, site in enumerate(sites):
        fp_data[site].attrs["Domain"] = fp_dict["domain"]
            
    return fp_data, sites


def biased_observations(data_dict: dict,
                        sitenames: list,
                        calib_tank_bias: dict, 
                        calib_tank_uncert: int,
                       ):
    """
    Function for applying calibration biases 
    and uncertainties to pseudo-observations.
    Calculates basis functions for biased observations 
    -----------------------------------------------
    Args:
        data_dict (dict):
            Output dictionary from "get_co2_data"
        sitenames (list):
            List of sites
        calib_tank_bias (list):
            Systematic calibration bias to apply to 
            pseudo-observations at each site
        calib_tank_uncert (list):
            List of uncertainties to apply to 
            pseudo-observations
    
    Returns
        fp_all (dict) with basis functions calculated 
        from data. 
    """
    fp_all = copy.copy(data_dict)

    for i, site in enumerate(sitenames):
        # Create biased pseudo-observations
        N = len(fp_all[site]["mf_mod_high_res"].values)
        calib_offset = np.random.normal(loc=calib_tank_bias[site], 
                                        scale=calib_tank_uncert, 
                                        size=N,
                                       )
        fp_all[site]["mf_pseudo"] = (("time"), fp_all[site]["mf_mod_high_res"].values + fp_all[site]["bc_mod"].values + calib_offset)

    return fp_all


def analytical_inversion(time,
                         mf_obs_all,
                         sens_all,
                         prior_sf,
                         prior_uncert,
                         model_data_uncert,
                        ):
    """
    -------- Analytical Inverse Model --------
    Analytical Bayesian Inverse model that 
    assumes variables are Normally distributed 

    Args:
        mf_obs_all
    
    """
    y = np.transpose(np.matrix(mf_obs_all))
    h = np.transpose(np.matrix(sens_all))
    n = y.shape[0]
    m = h.shape[1]
    
    # A priori scaling value
    if prior_sf is not None:
        sf_p = np.zeros([m, n]) + prior_sf
    else:
        sf_p = np.ones([m, n])
        
    # A priori uncertainty
    if prior_uncert is None:
        q = np.diag(np.ones(m))     
    elif type(prior_uncert) in [float, np.float64, int]:
        q = np.diag(np.ones(m)) * prior_uncert
    elif len(prior_uncert.shape) == 1:
        q = np.diag(prior_uncert) 
    else:
        q = prior_uncert.mean(axis = 2)
    
    # Model-data uncertainty
    if model_data_uncert is None:
        r = np.diag(np.ones(n))
    elif type(model_data_uncert) in [float, np.float64, int]:
        r = np.diag(np.ones(n)) * model_data_uncert
    else:
        r = np.diag(model_data_uncert)

    # sf_p: prior scaling factor
    # q: prior uncertainty
    # h: sensitivity (footprints x fluxes)
    # r: model-data uncertainty 
    # y: observations

    posterior = sf_p + (q @ h.T) @ np.linalg.inv(h @ q @ h.T + r) @ (y - h @ sf_p)
    post_cov  = q - (q @ h.T) @ np.linalg.inv(h @ q @ h.T + r) @ h @ q
    
    mf_prior     = h @ sf_p
    mf_posterior = h @ posterior

    results = {"time": time,
               "posterior_scale_factor": posterior,
               "posterior_scale_factor_cov": post_cov,
               "mf_prior": np.squeeze(mf_prior[:,0].T),
               "mf_posterior": np.squeeze(mf_posterior[:,0].T),
              }
    
    return results

def postprocessing_inversion_output(results: dict,
                                    countryfile: str, 
                                    nit: int,
                                    fp_all: dict,
                                   ):
    """
    Post-process the analytical inverse
    model outputs 
    ---------------------------------------------
    Args:
    results (dict):
        Dictionary containing analytical inversion 
        output
                
    countryfile (path):
        Path to countrfile with country region definitions
        
    nit (int)
        Number of runs 
    ---------------------------------------------
    Returns:
    cntryvalues:
        Country specific posterior emissions
    cntryprior:
        Country specific prior emissions
    cntrynames:
        Country names corresponding to posterior/
        prior country emissions
    apriorflux:
        A priori flux map
    fluxmap_post:
        Posterior flux map    
    """

    if countryfile is None:
        countryfile = "/user/work/wz22079/country_EUROPE_EEZ_PARIS_gapfilled.nc"
    
    time = results["time"]
    posterior_sf_mu = results["posterior_scale_factor"]
    posterior_sf_cov = results["posterior_scale_factor_cov"]
    mf_prior = results["mf_prior"]
    mf_posterior = results["mf_posterior"]

    # Get sitenames
    sitenames = []
    for key in fp_all.keys():
        if key[0] != '.':
            sitenames.append(key)
    
    # Get parameters for output file
    nit = nit
    nx = fp_all[sitenames[0]].H.shape[0]
    ny = len(mf_posterior)
    nbc = fp_all[sitenames[0]].H_bc.shape[0]
    nmeasure = np.arange(ny)
    nparam = np.arange(nx)
    nBC = np.arange(nbc)
    sitenum = np.arange(len(sitenames))
    
    Ymod = mf_posterior
    Yapriori = mf_prior
    
    lon = fp_all[sitenames[0]].lon.values
    lat = fp_all[sitenames[0]].lat.values
    site_lat = np.zeros(len(sitenames))
    site_lon = np.zeros(len(sitenames))
    for si, site in enumerate(sitenames):
        site_lat[si] = fp_all[site].release_lat.values[0]
        site_lon[si] = fp_all[site].release_lon.values[0]
        
    bfds = np.squeeze(fp_all[".basis"])
    bfarray = bfds.values-1
    
    nbasis = []
    for i in range(bfds.shape[0]):
        nbasis.append(int(np.nanmax(bfarray[i])))

    # Calculate mean posterior scale map and flux field
    scalemap = np.zeros_like(bfds.values)
    count = 0
    posterior_sf_mu_sector = []

    for fsector in range(scalemap.shape[0]):
        for npm in range(0, nbasis[fsector]):
            scalemap[fsector, bfds[fsector,:,:] == npm+1] = np.mean(posterior_sf_mu[npm+count, :])
            posterior_sf_mu_sector.append([posterior_sf_mu[npm+count, :]]) 
        count+=nbasis[fsector]

    aprioriflux_t = np.zeros_like(bfds.values)
    for i, key in enumerate(fp_all[".flux"].keys()):
        aprioriflux_t[i,:,:] = np.median(fp_all[".flux"][key].data["flux"].values, axis=2)
    
    # aprioriflux = np.median(aprioriflux_t["priorflux"], axis = 2)
    fluxmap_post = aprioriflux_t * scalemap

    area = utils.areagrid(lat, lon)
    c_object = utils.get_country("EUROPE", country_file=countryfile)
    
    cntryds = xr.Dataset({"country": (["lat","lon"], c_object.country),
                          "name": (["ncountries"], c_object.name) },
                         coords = {"lat": (c_object.lat),         
                                   "lon": (c_object.lon)})
    
    cntrynames = cntryds.name.values
    cntrygrid = cntryds.country.values
    
    cntryvalues = np.zeros((len(nbasis), len(cntrynames)))
    cntryprior = np.zeros((len(nbasis), len(cntrynames)))
    molarmass = convert.molar_mass("co2")

    unit_factor = convert.prefix(None)

    aprioriflux_tot = np.sum(aprioriflux_t, axis=0)

    for fsec in range(len(nbasis)):
        count = 0
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = 0
            cntrytotprior = 0

            for bf in range(int(np.max(bfarray[fsec,:,:]))+1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray[fsec,:,:]==bf)
            
                cntrytottrace += np.sum(area[bothinds].ravel() * aprioriflux_t[fsec, bothinds].ravel() *
                                        3600*24*365*molarmass) * posterior_sf_mu[count:nbasis[fsec]+count+1, :]/unit_factor
            
                cntrytotprior += np.sum(area[bothinds].ravel()*aprioriflux_t[fsec, bothinds].ravel()*
                                        3600*24*365*molarmass)/unit_factor

            cntryvalues[fsec, ci] = np.mean(cntrytottrace)
            cntryprior[fsec, ci] = cntrytotprior
        count+= nbasis[fsec]+1
    
    return cntryvalues, cntryprior, cntrynames, aprioriflux_t, fluxmap_post





def run_biased_inversions(nruns: int,
                          use_bc: bool,
                          sigma_freq: str,
                          outputname: str,
                          outputpath: str,
                          calib_tank_bias: float,
                          calib_tank_uncert: float,
                         ):
    """
    Function to run biased CO2 inversions
    -----------------------------------------------
    Args:
        nruns (int):
            No. of inversions to perform
        use_bc (bool):
            Option to run inversions with boundary conditions data
        outputname (str):
            Name used for saving any output files
        outputpath (str):
            Path to save output file
        calib_tank_bias (float):
            Systematic bias to apply to calibration tanks
        calib_tank_uncert (float):
            Uncertainty (1sigma) to apply to calibration biases 
            
    """
    if type(nruns)!= int:
        nruns = int(nruns)
        
    # Get input parameters
    obs_dict, flux_dict, fp_dict, bc_dict, basis_dict, mcmc_inputs_dict = inputs()
    
    # Get data 
    fp_data, sites = get_co2_data(use_bc=use_bc,
                                  outputname=outputname,
                                  outputpath=outputpath,
                                 )
    nsites = len(sites)

    uk_priors_emi = []
    uk_posteriors_emi = []
    calib_offsets_nruns = []
    mf_priors = []
    mf_posteriors = []
    
    for i in range(int(nruns)):
        print(f"####### RUNNNING INVERSION {i} #######")
        tank_offset = [0.0, calib_tank_bias]

        # Randomly assign a calibration bias to tank at each site 
        tank_allocation = np.random.randint(0, 2, nsites)

        # Using an uncertainty of 1sigma = calib_tank_uncert
        calib_tank_bias_dict = {}
        for j, site in enumerate(sites):
            calib_tank_bias_dict[site] = tank_offset[tank_allocation[j]]

        # Create biased pseudo-observations 
        fp_biased = biased_observations(data_dict=fp_data,
                                        sitenames=sites,
                                        calib_tank_bias=calib_tank_bias_dict,
                                        calib_tank_uncert=calib_tank_uncert,
                                       )

        # Prepare data for inversions 
        error = np.zeros(0)    
        Hbc = np.zeros(0)    
        Hx = np.zeros(0)    
        Y = np.zeros(0)    
        siteindicator = np.zeros(0)
        
        for j, site in enumerate(sites):
            # Select variables to drops NaNs from 
            drop_vars = []
            for var in ["H", "H_bc", "mf", "mf_variability", "mf_repeatability", "mf_pseudo", "mf_mod_high_res"]:
                if var in fp_biased[site].data_vars:
                    drop_vars.append(var)
                        
            # PyMC does not like NaNs, so drop them for the variables used below
            fp_biased[site] = fp_biased[site].dropna("time", subset=drop_vars)
        
            # Propagate mole fraction observational uncertainties for repeatability and variability
            # these terms should be added in quadrature
            myerror = np.zeros(0)
            if "mf_repeatability" in fp_biased[site]:
                if len(myerror)==0:
                    myerror = np.concatenate((myerror, fp_biased[site]["mf_repeatability"].values**2))                        
                elif len(myerror) == len(fp_biased[site]["mf_repeatability"].values):
                    myerror += fp_biased[site]["mf_repeatability"].values**2
                else:
                    raise(f"Error array length does not match length of mole fraction repeatability values for {site}.")
                    
            if "mf_variability" in fp_biased[site]:
                if len(myerror)==0:
                    myerror = np.concatenate((myerror, fp_biased[site]["mf_variability"].values**2))
                elif len(myerror) == len(fp_biased[site]["mf_variability"].values):
                    myerror += fp_biased[site]["mf_variability"].values**2
                else:
                    raise(f"Error array length does not match length of mole fraction variability values for {site}.")  
            error = np.concatenate((error, np.sqrt(myerror)))


            # Concatenate observational mole fractions for each site to Y 
            Y = np.concatenate((Y, fp_biased[site]["mf_pseudo"].values))
            siteindicator = np.concatenate((siteindicator, np.ones_like(fp_biased[site]["mf_pseudo"].values) * j))
                
            if j == 0:
                Ytime = fp_biased[site]["time"].values
            else:
                Ytime = np.concatenate((Ytime, fp_biased[site]["time"].values))
            
            if use_bc is True:
                bc_freq = bc_dict["bc_freq"]
            
                if bc_freq == "monthly":
                    Hmbc = setup.monthly_bcs(obs_dict["start_date"], 
                                             obs_dict["end_date"], 
                                             site, 
                                             fp_biased,
                                            )
                elif bc_freq is None:
                    Hmbc = fp_biased[site]["H_bc"].values
                else:
                    Hmbc = setup.create_bc_sensitivity(obs_dict["start_date"], 
                                                       obs_dict["end_date"], 
                                                       site, 
                                                       fp_biased,
                                                       bc_freq,
                                                      )
            elif use_bc is False:
                Hmbc = np.zeros(0)
            
            if j == 0:
                Hbc = np.copy(Hmbc)
                Hx = fp_biased[site]["H"].values
            else:
                Hbc = np.hstack((Hbc, Hmbc))
                Hx = np.hstack((Hx, fp_biased[site]["H"].values))

        # Mask source regions in Hx
        basis_region_mask = np.zeros_like(fp_biased[site]["region"].values)
        count = 0 
        for emi in flux_dict["source"]:
            count += 1
            for j in range(len(fp_biased[site]["region"].values)):
                if emi in fp_biased[site]["region"].values[j]:
                    basis_region_mask[j] = count
        basis_region_mask = basis_region_mask.astype(int)

        sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)

        YBC = np.sum(Hbc, axis=0) * 1e6

        results = analytical_inversion(time=Ytime,
                                       mf_obs_all=Y-YBC,
                                       sens_all=Hx,
                                       prior_sf=1.0,
                                       prior_uncert=0.5,
                                       model_data_uncert=error,
                                      )

        results["mf_prior"] = YBC + results["mf_prior"]
        results["mf_posterior"] = YBC + results["mf_posterior"]

        cntryvalues, cntryprior, cntrynames, aprioriflux, fluxmap_post = postprocessing_inversion_output(results,
                                                                                                         countryfile=None, 
                                                                                                         nit=nruns,
                                                                                                         fp_all=fp_biased,
                                                                                                        )


        uk_priors_emi.append(np.sum(cntryprior[:,15]/1e12))
        uk_posteriors_emi.append(np.sum(cntryvalues[:,15]/1e12))
        mf_priors.append(np.squeeze(np.asarray(results['mf_prior'][0,:])))
        mf_posteriors.append(np.squeeze(np.asarray(results['mf_posterior'][0,:])))
        # calib_offsets_nruns.append(calib_tank_bias_dict)


    data_vars = {"UK_prior_emissions": (["nruns"], np.array(uk_priors_emi)),
                 "UK_posterior_emissions": (["nruns"], np.array(uk_posteriors_emi)),
                 "mf_prior": (["nruns", "siteindicator"], np.array(mf_priors)),
                 "mf_posterior": (["nruns", "siteindicator"], np.array(mf_posteriors)),
                }
    
    coords = {"nruns": (["nruns"], np.arange(1, nruns+1, 1)),
              "siteindicator": (["siteindicator"], siteindicator),
              "sitenames": (["sitenames"], sites),
             
             }
    outds = xr.Dataset(data_vars, coords=coords)

    outds.UK_prior_emissions.attrs["unit"] = "Tg CO2"
    outds.UK_posterior_emissions.attrs["unit"] = "Tg CO2"

    outds.attrs["bias"] = calib_tank_bias
    outds.attrs["uncert"] = calib_tank_uncert

    savepath = "/user/work/wz22079/projects/CO2/NPL-calibration/results/"
    savename = f"{outputname}-CO2-CBW_MHD_BSD_HFD_TAC_RGL_WAO-bias_{calib_tank_bias}pmm-uncert_{calib_tank_uncert}-r{nruns}.nc"

    output_filename = os.path.join(savepath, savename)
    
    outds.to_netcdf(output_filename, mode="w")

    return uk_priors_emi, uk_posteriors_emi


def main():
    #calib_tank_bias_list = [0.01, 0.025, 0.050, 0.075, 0.10, 0.15, 0.20]
    calib_tank_bias_list = [0.01]
    for bias in calib_tank_bias_list:
        uk_priors_emi, uk_posteriors_emi = run_biased_inversions(nruns=100,
                                                                 use_bc=True,
                                                                 sigma_freq=None,
                                                                 outputname="NPL-inversions",
                                                                 outputpath="/user/work/wz22079/",
                                                                 calib_tank_bias=bias,
                                                                 calib_tank_uncert=0.5,
                                                                )


if __name__ == "__main__":
    main()
