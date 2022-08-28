# coding: utf-8

# parse arguments
import argparse
parser = argparse.ArgumentParser(description="Run transit fitting.")
parser.add_argument("-in_file","--input_file", type=str, help= "Filename for text file in this directory containing list of star names to run on.") 
parser.add_argument("-start", "--start", type=int, help="Index of planet to start on.")
parser.add_argument("-stop", "--stop", type=int, help="Index of planet to stop on.")
parser.add_argument("-out_dir","--output_directory", type=str, help= "") 
parser.add_argument("-dont_skip_shallow",  help="Do NOT skip planets whose expected depth is less than twice the ligthcurve mean flux error.", action = 'store_true') 
parser.add_argument("-siblings",  help="This run is for additional sibling planets not in the dataset.", action = 'store_true')# default is false (ie. for regular target set planets)
parser.add_argument("-const_depth",  help="Do NOT fit for depth (use expected value)", action = 'store_true') # default is false (ie. skip them)
args = parser.parse_args()
in_file = args.input_file
out_dir = args.output_directory
skip_shallow = not args.dont_skip_shallow
to_fit_depth = not args.const_depth
start = args.start
stop = args.stop
siblings = args.siblings

# import packages
import batman
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import lightkurve as lk
from scipy.optimize import curve_fit
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from lmfit import Model
from lmfit import Parameter
import pandas as pd
import csv
import os
import sys
import datetime
from astroquery.simbad import Simbad 
import traceback as tb

# NASA exoplanet archive table
NASA_exoplanet_table = pd.read_csv('NASA_exoplanet_archive.csv', skiprows = 292)

# Stacked/mean fit parameters (NOT mean of individual fitting)
#mean_fit_results = pd.read_csv('folded_fit_params.csv')

if not siblings:
  mean_fit_results = pd.read_csv('mean_fit_params_manually_updated.csv')
if siblings:
  mean_fit_results = pd.read_csv('mean_fit_params_siblings_6-24-22.csv')

# read in mean fit depths
depth_err_ratios = pd.read_csv('KOI_depth_err_ratios_updated.csv')

# read in flux SD
flux_SD_table = pd.read_csv('mean_rolling_SD.csv')

# define an exception for if expected transit depth is within flux uncertainty is found in the spectrum
class Expected_Depth_Exception(Exception):
    pass

# remove trends in baseline flux data
def linear_func(x, a, b):
  return a*x + b

def stitch(lc_collection): 

    tot_time = np.zeros(0) # to store concatenated arrays
    tot_flux = np.zeros(0)
    tot_flux_err = np.zeros(0)
    tot_qual = np.zeros(0)
    for i in range(len(lc_collection)):
        lc = lc_collection[i]
        flux = lc.flux.value
        time = lc.time.value
        flux_err = lc.flux_err.value
        qual = lc.quality
        rel_flux_err = flux_err/flux
        nan_mask = np.invert(np.isnan(flux))
        
        # Fit and remove linear trend
        popt, pcov = curve_fit(linear_func, time[nan_mask], flux[nan_mask])
        linear_trend = linear_func(time, *popt) # evaluate over the whole interval
        norm = flux / linear_trend
        
        tot_time = np.hstack([tot_time, time])
        tot_flux = np.hstack([tot_flux, norm])
        tot_flux_err = np.hstack([tot_flux_err, rel_flux_err])
        tot_qual = np.hstack([tot_qual, qual])
              
    return tot_time, tot_flux, tot_flux_err, tot_qual

def make_fits(planet_name, shift_value, plots_dir, flag_plot_path, plot_lightcurve, verbosity = 0, plot_verbosity = 1, plot_skipped = 0, 
              plot_flags = 0 ):

  # get properties
  if planet_name.startswith('kepler-') or planet_name.startswith('Kepler-'):
    planet_name = planet_name.replace('k','K')
    star_name = planet_name[:-1].replace(' ', '')
    obs_type = planet_name.split('-')[0]
    
  else:
    star_name = input('Enter star name: ')
    obs_type = input('Enter whether this is a \'kepler\' or \'TESS\' observation: ')
    

  print('Fitting ' + planet_name + ': ')
 
  query_name = planet_name
  if query_name == 'Kepler-1 b':  query_name = 'TrES-2 b'
  if query_name == 'Kepler-458 b': query_name = 'KIC 9663113 b'
  if query_name == 'Kepler-324 d': query_name = 'KOI-1831 d'    
  if query_name == 'Kepler-1703 b':  query_name = 'KOI-3503 b'
  if query_name == 'Kepler-968 d':  query_name = 'KOI-1833 d'
  if query_name == 'Kepler-460 b':  query_name = 'KIC 5437945 b'
  if query_name == 'Kepler-86 b':  query_name = 'PH2 b'
  if query_name == 'Kepler-1703 c':  query_name = 'KOI-3503 c' 

  print('querying with name: ' + query_name)

  planet =  NASA_exoplanet_table[NASA_exoplanet_table['pl_name'] == query_name]

  if planet.empty:
      print('tying alternate names.')
      result_table = Simbad.query_objectids(planet_name)
      alt_names = result_table.to_pandas()
      alt_names = list(alt_names.iloc[:,0].str.decode('utf-8')) # gets rid of weird formatting
      for new_name in alt_names:
          #new_name = new_name[0].split(' ')[-1]
          if new_name[-1].islower():
              new_name = new_name[:-1] + ' ' + new_name[-1]
       #   try:
          planet =  NASA_exoplanet_table[NASA_exoplanet_table['pl_name'] == new_name]
          if not planet.empty:
                break 
     #   except: continue
            
  period_guess = np.nanmedian(planet['pl_orbper']) # Planet period [Days]
  t0_guess = np.nanmedian(planet['pl_tranmid']) - 2454833 # Transit midpoint [Days]
  R_p = np.nanmedian(planet['pl_rade']) * 0.0091577 # Planet radius [Ro]
  R_star = np.nanmedian(planet['st_rad'])  # Star radius [Ro]
  R_star_AU = R_star * 0.00465047
  impact_parameter_guess = np.nanmedian(planet['pl_imppar']) # impact parameter 
  duration_guess = np.nanmedian(planet['pl_trandur']) / 24 # Planet transit duration [Days]
  semi_maj_ax_guess = np.nanmedian(planet['pl_orbsmax']) / R_star_AU # semi-major axis [Stellar Radii]
  inc_guess = np.nanmedian(planet['pl_orbincl']) # inclination [deg]
  arg_of_periastron_guess = np.nanmedian(planet['pl_orblper']) # arguement/longitude of periastron/periapse [deg]
  if np.isnan(arg_of_periastron_guess):
    arg_of_periastron_guess = 0 # Possibly this is an ok default 
  else:
    arg_of_periastron_guess = arg_of_periastron_guess
  ecc_guess =  np.nanmedian(planet['pl_orbeccen']) # eccentricity
  RpRs_guess = R_p/R_star # Planet star radius ratio = sqrt(transit depth)
  u_guess = [0.5, 0.1, 0.1, -0.1] # Possibly this is an ok default    
  u1_guess = u_guess[0]; u2_guess = u_guess[1]
  expected_depth = (RpRs_guess**2) * np.cos(impact_parameter_guess)
    
  

  expected_params = {'period': period_guess, 't0':t0_guess ,  'impact_parameter':impact_parameter_guess, 'duration':duration_guess, 
                     'semi_maj_ax': semi_maj_ax_guess, 'inclination':inc_guess, 'arg_of_periastron':arg_of_periastron_guess, 
                     'eccentricity':ecc_guess, 'RpRs':RpRs_guess, 'u1':u1_guess, 'u2':u2_guess, 'depth':expected_depth}
  #expected_params = {'t0':t0_guess , 'duration':duration_guess}
  
  mean_fit_params = mean_fit_results[mean_fit_results['Name' ] == planet_name]

  
  #mean_fit_depth = depth_err_ratios[depth_err_ratios['NAME'] == planet_name]['MEAN_FIT_DEPTH'] # THIS NEEDS TO BE UPDATED WHEN THE FIT PARAMS ARE
  mean_fit_depth = mean_fit_params['depth']

  #if verbosity > 0:
  #  print('period: ' + str(np.round(period_guess,3)))
  #  print('duration: ' + str(np.round(duration_guess,3)))
  #  print('t0: ' + str(np.round(t0_guess,3)))
  #  print('RpRs: ' + str(np.round(RpRs_guess,3)))
  #  print('Depth: ' + str(np.round(expected_depth,5)))
  #  print('impact parameter: ' + str(np.round(impact_parameter_guess,3)))
  #  print('w: ' + str(np.round(arg_of_periastron_guess,3)))

  # search for and download all lightcurves
  search_result = lk.search_lightcurve(planet_name, author=obs_type, cadence='long') # WAS star_name -- does this work better?
  lc_collection = search_result.download_all()
  lc = lc_collection.stitch() #.remove_outliers() # it turns out removing outliers can remove transits as well! As can flattening with a poor window choice. 
  crowdsap = lc.meta['CROWDSAP'] # A measure of contamination from nearby stars
  #flux = lc.flux.value           # Flux [relative units]
  #time = lc.time.value           # Time  [BKJD days]
  #quality = lc.quality           # Quality flags
  #flux_err = lc.flux_err.value   # Uncertainty in flux
  time, flux, flux_err, quality = stitch(lc_collection)
  dt = np.median(np.diff(time))  # [Days] Usual timestep size (should match cadence, except for example missing data)
  #flux_SD = np.std(flux)         # Standard deviation in flux
  mean_flux_err = np.nanmean(flux_err)  # Mean error in flux 
    
    #return
    # TO DO:
    # once we are looping over many lightcurves, add a continue statement here 
    # and add a log varaible to record which are skipped 

  # stitch together, flatten, and normalize all observations
  #fwindow_length = int(duration_guess * 4) # int(1/dt)   # ideally this would be 3-4 * transit duration, but duration is not known until transit is modelled. So for now we consider 24hrs.
  #if window_length % 2 == 0: # length must be odd for built in lightkurve smoothing function
  #    window_length += 1
  #lc = lc.flatten(window_length) # shouldn't we do this BEFORE defining the flux, flux_err, and time variables?


  # JUST FOR FUN, NOT USED:
  # plot a phase folded lightcurve
  #periods = np.linspace(1, 20, 10000)
  # Create a BLSPeriodogram
  #bls = lc.to_periodogram(method='bls', period=periods, frequency_factor=500)
  #planet_b_period = bls.period_at_max_power; planet_b_t0 = bls.transit_time_at_max_power; planet_b_dur = bls.duration_at_max_power
  #folded_lc = lc.fold(period=planet_b_period, epoch_time=planet_b_t0)
  #plt.figure(figsize = [15,5])
  #plt.plot(folded_lc.phase.value, folded_lc.flux.value, '.', label = 'Flux', markersize = 4)
  #ax = folded_lc.scatter()
  #ax.set_xlim(-5, 5);
  #plt.margins(x=0.01) 
  #plt.ylabel('Normalized relative flux', fontsize = 16)
  #plt.xlabel('Phase [JD]', fontsize = 16)
  #plt.xlim([-0.5,0.5])
  #plt.legend(fontsize = 16)
  #plt.figure(figsize = [15,5])
  #plt.plot(folded_lc.phase, folded_lc, '.', label = 'Flux', markersize = 4)
  #plt.margins(x=0.01)
  #plt.ylabel('Normalized relative flux', fontsize = 12)
  #plt.xlabel('Phase [JD]', fontsize = 12)
  #plt.legend()

  # mask out transits of other planets in the system
  try:
      system_results = NasaExoplanetArchive.query_object(star_name)
      planets = system_results['pl_letter'].pformat()[2:]
      unique_planets = set([planet for planet in list(set(planets)) if (not planet.startswith('Length')) and (not planet.endswith('.'))])
      #unique_planets.remove('Length = 24 rows')
      system_periods = list(system_results['pl_orbper'])
      system_t0 = list(system_results['pl_tranmid']) 
      system_dur = list(system_results['pl_trandur']) 
      no_mask_flux = np.copy(flux)
      no_mask_time = np.copy(time)
      no_mask_quality = np.copy(quality)
      mask = np.array([False]) * len(flux)
      bls = lc.to_periodogram(method='bls', period=np.linspace(1,50), frequency_factor=500) # need this to use lightkurve's masking function
      for planet in unique_planets: 
        if planet_name[-1] != planet[-1]:
          print('masking other transits.')
          unique_period = np.nanmedian(system_periods[planets.index(planet)]).value
          unique_t0 = np.nanmedian(system_t0[planets.index(planet)]).value - 2454833
          if np.isnan(np.nanmedian(system_dur[planets.index(planet)])):
            continue
          unique_duration = np.nanmedian(system_dur[planets.index(planet)]).value / 24
          planet_mask = bls.get_transit_mask(period = unique_period, transit_time = unique_t0, duration = unique_duration)
          mask = np.logical_or(mask,planet_mask)  
  except:
      system_results = NASA_exoplanet_table[NASA_exoplanet_table['hostname'] == star_name]
      planets = list(system_results['pl_letter'])
      unique_planets = set([planet for planet in list(set(planets)) if (not planet.startswith('Length')) and (not planet.endswith('.'))])
      #unique_planets.remove('Length = 24 rows')
      system_periods = list(system_results['pl_orbper'])
      system_t0 = list(system_results['pl_tranmid']) 
      system_dur = list(system_results['pl_trandur']) 
      no_mask_flux = np.copy(flux)
      no_mask_flux_err = np.copy(flux_err)
      no_mask_time = np.copy(time)
      no_mask_quality = np.copy(quality)
      mask = np.zeros(len(flux)).astype(bool)
      bls = lc.to_periodogram(method='bls', period=np.linspace(1,50), frequency_factor=500) # need this to use lightkurve's masking function
      for planet in unique_planets: 
        if planet_name[-1] != planet[-1]:
          unique_period = np.nanmedian(system_periods[planets.index(planet)])
          unique_t0 = np.nanmedian(system_t0[planets.index(planet)]) - 2454833
          if np.isnan(np.nanmedian(system_dur[planets.index(planet)])):
            continue
          unique_duration = np.nanmedian(system_dur[planets.index(planet)]) / 24
          #planet_mask = bls.get_transit_mask(period = unique_period, transit_time = unique_t0, duration = unique_duration)
          # in the above line, bls.get_transit_mask seems to remove outliers, thus often resulting in a different length mask than the lighcurve itself
          planet_mask = lc.create_transit_mask(period=float(unique_period), transit_time=float(unique_t0), duration=float(unique_duration))
          mask = np.logical_or(mask,planet_mask)  # errors out on this line for some stars! lc and flux are not same length?
      
  
  #transit_masked_flux = np.copy(flux)[mask] # all the points that are masked due to another transit
  transit_masked_time = time[mask] # all the points that are masked due to another transit
  flux = flux[np.invert(mask)]
  flux_err = flux_err[np.invert(mask)]
  time = time[np.invert(mask)]

  quality_no_mask = np.copy(quality)
  quality = quality[np.invert(mask)]


  # assesss quality flags and remove bad data
  # try a more lenient quality cut
  no_quality_cut = (quality >= 0)
  quality_cut_lenient = (quality != 16) * (quality != 128) * (quality != 2048) 
  quality_cut_medium = (quality != 1) * (quality != 4) * (quality != 16) * (quality != 32) * (quality != 128) * (quality != 256) * (quality != 2048) * (quality != 32768)
  quality_cut_aggressive = (quality == 0) # this is the original, default cut
  good_data_flag = quality_cut_aggressive * (flux_err > 0) * (np.isfinite(time)) * (np.isfinite(flux)) * (np.isfinite(flux_err)) 
  bad_time = time[np.invert(good_data_flag)]
  bad_flux = flux[np.invert(good_data_flag)]     
  bad_flux_err = flux_err[np.invert(good_data_flag)]     
  time = time[good_data_flag]
  flux = flux[good_data_flag]
  flux_err = flux_err[good_data_flag]
  
  # calculate standard deviation in the flux (testing this as a depth threshold)
  #df = pd.DataFrame(flux)
  #med_flux_SD = np.nanmedian(df.rolling(10).std())
  #print('3 * flux_SD: ' + str(3 * med_flux_SD))   
    
  # get mean rolling flux SD
  try:
      med_flux_SD = list(flux_SD_table[flux_SD_table['Name'] == planet_name]['Mean_rolling_SD'])[0]
  except IndexError: # the planet wasn't in the overall list of ~2360 targets, and so has no entry in the SD table
      med_flux_SD = np.nanmedian(pd.DataFrame(flux).rolling(10).std())
        
    
  # Warning!
  print('mean_fit_depth:' + str(mean_fit_depth))
  print('mean_flux_err:' + str(mean_flux_err))
  print('flux_sd: ' + str(med_flux_SD))
  if ((float(mean_fit_depth) <= 2*mean_flux_err) or float(mean_fit_depth) <= 3 * med_flux_SD):
    print('WARNING: ')
    print('    Expected transit depth is less than twice the average uncertainty in flux OR less than twice the mean rolling flux SD ' +
          'and thus statements about missing transits in this lightcurve ' +
          'are not statistically relevent. Proceed at your own risk!')
    if skip_shallow:
        raise Expected_Depth_Exception('Expected transit depth is less than twice the average uncertainty in flux  OR less than twice the mean rolling flux SD and thus statements about missing transits in this lightcurve are not statistically relevent, and thus statements about missing transits in this lightcurve  would not be not meaningful.')


  # define data that is masked for any reason (and not used in fits)
  good_data_no_mask = (no_mask_quality == 0) * (no_mask_flux_err > 0) * (np.isfinite(no_mask_time)) * (np.isfinite(no_mask_flux)) * (np.isfinite(no_mask_flux_err)) 
  masked_flux = no_mask_flux[mask | np.invert(good_data_no_mask)]
  masked_flux_err = no_mask_flux_err[mask | np.invert(good_data_no_mask)]
  masked_time = no_mask_time[mask | np.invert(good_data_no_mask)]

  #num_transits = int(np.floor((time[-1] - t0_guess) / period_guess))
  #expected_midpoints = np.arange(t0_guess, t0_guess + (num_transits) * period_guess, period_guess)
  num_transits = int(np.floor((time[-1] - t0_guess) / period_guess)) + 1
  expected_midpoints = np.linspace(t0_guess, t0_guess + (num_transits - 1) * period_guess, num_transits)
  if shift_value > 0:
    shifted = False
    shift_value = 1/3 #period_guess/3   # period_guess/3 # --> for testing regions with no transit
                    # = 1/8 for testing shifting
                    # also can do 30min, 1hr, 2hr ect to simply shift transit
    time = time + shift_value; shifted = True; # shift time 

  # get cadence information
  cadence_type = lc.meta['OBSMODE'] 
  if obs_type == 'Kepler' or obs_type == 'kepler':
      if cadence_type == 'long cadence':
          cadence = 1764 # sec
      if cadence_type == 'short cadence':
          cadence = 58.85 # sec   
          
  if obs_type == 'TESS':
      print('TESS CADENCE INFO NOT YET IMPLEMENTED!')

  # plot the entire lightcurve
  if plot_lightcurve:
    #expected_midpoints = np.arange(t0_guess, t0_guess + (num_transits) * period_guess, period_guess)
    plt.figure(figsize = [15,5])
    plt.plot(time, flux, '.', label = 'Flux', markersize = 4)
    plt.plot(bad_time, bad_flux, '.', label = 'Quality flagged', markersize = 3)
    plt.plot(no_mask_time[mask], no_mask_flux[mask], '.', label = 'Masked flux (other planet transists)', markersize = 3)
    plt.errorbar(time, flux, yerr = flux_err, color = 'gray', alpha = 0.5, ls = 'None', label = 'Flux error')
    plt.vlines(expected_midpoints, ymin = np.nanmin(flux), ymax = np.nanmax(flux), linewidth = 1.5, color = 'gray', label = 'Expected transits')
    plt.margins(x=0.01)
    plt.ylabel('Normalized relative flux', fontsize = 16)
    plt.xlabel('Time [BKJD Days]', fontsize = 16)
    plt.legend(fontsize = 16)
    #plt.xlim([200,500]) # just for better visibility in this case
    plt.savefig(out_dir + '/' +  planet_name + '/Full_lightcurve.png')

  fit_window_size = 3    # Number of transit durations to fit on both sides of the expected transit midpoints 
  inner_window_size = 1  # Number of transit durations on both sides of expected midpoint to check for missing data
  expected_points_per_inner_window = np.floor((duration_guess * 24 * 60 * 60 * inner_window_size*2) / cadence)


  # define fit function
  def func(t, t0, period, RpRs, a, inc, ecc, w, u1, u2, low_bound, up_bound):

    # set up a transit object
    params = batman.TransitParams()
    params.t0 = t0                 # time of inferior conjunction
    params.per = period            # orbital period [days]
    params.rp = RpRs               # planet radius (in units of stellar radii) [Rp/R*]
    params.a = a                   # semi-major axis (in units of stellar radii)
    params.inc = inc               # orbital inclination (in degrees)
    params.ecc = ecc               # eccentricity
    params.w = w                   # longitude of periastron (in degrees) -- documentation says argument of periapse [deg]
    params.u = [u1, u2]            # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic" # limb darkening model

    # initialze model + calculate lightcurve
    #interval = (low_bound < time)*(time < up_bound) # interval to fit on
    #t = time[interval]                              # times to calculate at
    m = batman.TransitModel(params, t)              # initializes model
    model_flux = m.light_curve(params)              # calculates light curve

    #print(t
    return model_flux

  def make_fit(transit_idx, no_points, plot_verbosity, verbosity, plots_dir, flag_plot_path):

    
    # area around expected transit -- use nasa reported period not period from stacked fitting (stacked fit period reflects transit shape not periodicity well)
    low_bound = t0_guess + period_guess * transit_idx - duration_guess * fit_window_size # t0_guess + period_guess * transit_idx - period_guess * fit_window_size    
    up_bound = t0_guess + period_guess * transit_idx + duration_guess * fit_window_size # t0_guess + period_guess * transit_idx + period_guess * fit_window_size  
    
    # all intervals are ~same in time duration, but not in number of steps/pixels
    interval = (low_bound < time)*(time < up_bound)  
    mask_interval = (low_bound < masked_time)*(masked_time < up_bound) # for the masked data, to use in plotting
    transit_mask_interval = (low_bound < transit_masked_time)*(transit_masked_time < up_bound) # for the masked data, to use in plotting
    time_interval = np.max(time[time < up_bound]) -  np.min(time[time > low_bound]) 
    length = float(len(time[interval]))  
    if length == 0:
      no_points += [transit_idx]
    
    
    # for checking that datapoints immediatley around transit are not missing
    # becuase the location of the missing data within the transit window mattters
    inner_low_bound = t0_guess + period_guess * transit_idx - duration_guess * inner_window_size #t0_guess + period_guess * transit_idx - period_guess * inner_window_size    
    inner_up_bound = t0_guess + period_guess * transit_idx + duration_guess * inner_window_size #t0_guess + period_guess * transit_idx + period_guess * inner_window_size    
    
    # all intervals are ~same in time duration, but not in number of steps/pixels
    inner_interval = (inner_low_bound < time)*(time < inner_up_bound)
    outer_interval = interval*(np.invert(inner_interval)) # mask out the transit itself
    inner_time_interval = np.max(time[time < inner_up_bound]) -  np.min(time[time > inner_low_bound])
    inner_length = len(time[inner_interval])
    apply_thresholds = 1
    min_points_threshold = expected_points_per_inner_window * 0.70
    if apply_thresholds:
        if (inner_length < min_points_threshold):
            print('Could not model transit ' + str(transit_idx) + ' (due to missing data).')   
            if plot_verbosity > 0:
              plt.figure(figsize = [8,4])
              plt.plot(time[interval], flux[interval], '.', label = 'Flux data')
              plt.plot(masked_time[mask_interval], masked_flux[mask_interval], '.', color = 'C0', alpha = 0.25, label = 'Masked')#(quality)') 
              #if not len(transit_masked)
              transit_mask_time = transit_masked_time[transit_mask_interval]
              idxs = np.where(np.isin(time, transit_mask_time))
              #plt.plot(transit_mask_time, flux[idxs], '^', color = 'C0', alpha = 0.25, label = 'Masked (sibling transit)')    
              plt.axvline(t0_guess + period_guess * transit_idx, ls = '-.', color = 'black', alpha = 0.75, label = 'Expected midpoint')
              plt.errorbar(time[interval], flux[interval], yerr = flux_err[interval], color = 'gray', ls='none')#, label = 'Flux uncertainty',)
              plt.errorbar(masked_time[mask_interval], masked_flux[mask_interval], yerr = masked_flux_err[mask_interval], color = 'black', alpha = 0.25, ls='none')
              mid = t0_guess + period_guess * transit_idx
              plt.axvline(mid - duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25, label = 'Expected Ingress/Egress')
              plt.axvline(mid + duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25)
              plt.legend()
              plt.xlabel('Time [Days]')
              plt.ylabel('Relative Flux')
              plt.title(planet_name + ', transit ' + str(int(transit_idx)))
              ax = plt.gca()
              ax.ticklabel_format(useOffset=False)
              plt.savefig(plots_dir + planet_name + '_t' + str(transit_idx) + '_skipped.png')
            
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1
        
    # define sorted idxs (to ensure data is sorted by time)    
    sorted_idxs_interval = np.argsort(time[interval])
    sorted_idxs_outer_interval = np.argsort(time[outer_interval])
    sorted_idxs_mask_interval = np.argsort(masked_time[mask_interval])
    #sorted_idxs_transit_mask_interval = np.argsort(masked_time[transit_mask_interval])    

    
    # remove trends in baseline flux data
    #def linear_func(x, a, b):
    #  return a*x + b
    popt, pcov = curve_fit(linear_func, time[outer_interval][sorted_idxs_outer_interval], flux[outer_interval][sorted_idxs_outer_interval])
    linear_trend = linear_func(time[interval][sorted_idxs_interval], *popt) # evaluate over the whole interval
    flux_data = flux[interval][sorted_idxs_interval] - linear_trend + 1 # +1 to hold the median flux at 1
    flux_err_data = flux_err[interval][sorted_idxs_interval] #- linear_trend + 1 # +1 to hold the median flux at 1
    # and again for the masked ('bad') data, for plotting purposes

    # create model
    fmodel = Model(func)

    x = time[interval][sorted_idxs_interval]
    midpoint_guess = t0_guess + period_guess * transit_idx # t0_value 
    
    print(mean_fit_params)
    
    # perform initial brute force fit -- essentially this step is just to better determine the expected transit midpoint
    to_fit_depth = True
    if to_fit_depth:
        print('fitting includes depth fitting.')
        initial_result = fmodel.fit(data=flux_data, t=x,
          period = Parameter('period', value = float(mean_fit_params['period']), vary = False), # use stacked fit period value for shape
          a = Parameter('a', value = float(mean_fit_params['a']), vary = False),
          inc = Parameter('inc', value = float(mean_fit_params['inc']), vary = False),
          ecc = Parameter('ecc', value = float(mean_fit_params['ecc']), vary = False),
          w = Parameter('w', value = float(mean_fit_params['w']), vary = False),
          u1 = Parameter('u1', value = float(mean_fit_params['u1']), vary = False),
          u2 = Parameter('u2', value = float(mean_fit_params['u2']), vary = False),
          RpRs = Parameter('RpRs', value = float(mean_fit_params['RpRs']), vary = True, min = 0.001, max = 1),
          #RpRs = Parameter('RpRs', value = RpRs_guess , vary = False), # JUST FOR TESTING
          t0 = Parameter('t0', value = midpoint_guess, vary = True, min = low_bound, max = up_bound), # use nasa reported period value for midpoint
          #t0 = Parameter('t0', value = midpoint_guess , vary = False), # JUST FOR TESTING
          low_bound = Parameter('low_bound', value = low_bound, vary = False),
          up_bound = Parameter('up_bound', value = up_bound, vary = False),
          method = 'brute',
          calc_covar = True)
    #if not to_fit_depth: # For Evan Sneed
    #        print('fitting does not include depth fitting.')
    #        initial_result = fmodel.fit(data=flux_data, x=x,
    #          period = Parameter('period', value = period_guess, vary = False),
    #          a = Parameter('a', value = semi_maj_ax_guess, vary = False),
    #          inc = Parameter('inc', value = inc_guess, vary = False),
    #          ecc = Parameter('ecc', value = ecc_guess, vary = False),
    #          w = Parameter('w', value = arg_of_periastron_guess, vary = False),
    #          u1 = Parameter('u1', value = u1_guess, vary = False),
    #          u2 = Parameter('u2', value = u2_guess, vary = False),
    #          RpRs = Parameter('RpRs', value = RpRs_guess, vary = False),
    #          #RpRs = Parameter('RpRs', value = RpRs_guess , vary = False), # JUST FOR TESTING
    #          t0 = Parameter('t0', value = midpoint_guess, vary = True, min = low_bound, max = up_bound),
    #          #t0 = Parameter('t0', value = midpoint_guess , vary = False), # JUST FOR TESTING
    #          low_bound = Parameter('low_bound', value = low_bound, vary = False),
    #          up_bound = Parameter('up_bound', value = up_bound, vary = False),
    #          method = 'brute',
    #          calc_covar = True)     
            
    # update the expected transit midpoint to update the transit mask
    #new_low_bound =  initial_result.params['t0'] + period_guess * transit_idx - duration_guess * fit_window_size 
    #new_up_bound = initial_result.params['t0'] + period_guess * transit_idx + duration_guess * fit_window_size 
    #new_interval = (new_low_bound < time)*(time < new_up_bound) 
    new_inner_low_bound = initial_result.params['t0'] - duration_guess * inner_window_size 
    new_inner_up_bound = initial_result.params['t0'] + duration_guess * inner_window_size 
    new_inner_interval = (new_inner_low_bound < time)*(time < new_inner_up_bound)

    # update the linear baseline trend to remove
    new_outer_interval = interval*(np.invert(new_inner_interval)) # mask out the transit itself
    sorted_idxs_new_outer_interval = np.argsort(time[new_outer_interval])
    new_popt, new_pcov = curve_fit(linear_func, time[new_outer_interval][sorted_idxs_new_outer_interval], flux[new_outer_interval][sorted_idxs_new_outer_interval])
    new_linear_trend = linear_func(time[interval][sorted_idxs_interval], *new_popt) # evaluate over the whole interval
    updated_flux_data = flux[interval][sorted_idxs_interval] - new_linear_trend + 1 # +1 to hold the median flux at 1 
    shifted_flux = updated_flux_data
    shifted_flux_err = flux_err[interval][sorted_idxs_interval] #- new_linear_trend + 1 # +1 to hold the median flux at 1
    # and again for the masked ('bad') data, for plotting purposes

    masked_linear_trend = linear_func(masked_time[mask_interval][sorted_idxs_mask_interval], *popt) # evaluate over the whole interval
    shifted_masked_flux = masked_flux[mask_interval][sorted_idxs_mask_interval] - masked_linear_trend + 1 # +1 to hold the median flux at 1
    shifted_masked_flux_err = masked_flux_err[mask_interval][sorted_idxs_mask_interval] #- masked_linear_trend + 1 # +1 to hold the median flux at 1
  
    #np.set_printoptions(threshold=sys.maxsize)
    #print(time[interval][sorted_idxs_interval])
    #print(updated_flux_data)

    
    # perform second fit using updated baseline trend removal 
    result = fmodel.fit(data=updated_flux_data, t=x, params = initial_result.params, method = 'least_squares')

    
    # create the fit curve (evaluate over lightcurve time for calculating residuals)
    fit_curve = func(time[interval][sorted_idxs_interval], result.params['t0'], result.params['period'], result.params['RpRs'], 
                    result.params['a'], result.params['inc'], result.params['ecc'], result.params['w'], 
                    result.params['u1'], result.params['u2'], low_bound, up_bound)
    
    # evaluate over interpolated time for determing depth and for plotting clarity
    t = np.linspace(time[interval][0], time[interval][-1], 100) # time[interval][sorted_idxs_interval]
    fit_curve_plotting = func(t, result.params['t0'], result.params['period'], result.params['RpRs'], 
                    result.params['a'], result.params['inc'], result.params['ecc'], result.params['w'], 
                    result.params['u1'], result.params['u2'], low_bound, up_bound)
    
    # determine fit depth empirically 
    fit_depth = 1 - np.nanmin(fit_curve_plotting)
    
    expected_fit_curve = func(t, midpoint_guess, float(mean_fit_params['period']), float(mean_fit_params['RpRs']), 
                    float(mean_fit_params['a']), float(mean_fit_params['inc']), float(mean_fit_params['ecc']), float(mean_fit_params['w']), 
                    float(mean_fit_params['u1']), float(mean_fit_params['u2']), low_bound, up_bound)

    resids = flux_data - fit_curve
    RMSE = np.sqrt(np.sum(resids**2/length))
    chi_sq = np.sum((resids**2)/flux_err[interval]**2)
    k = 2 # only t0 and RpRs vary
    BIC = chi_sq/(length - k) + k*np.log(length) # bayesian information criterion
    
    #print('fit RpRs transit ' + str(transit_idx) + ': ' + str(result.params['RpRs']))
    #print('fit depth transit ' + str(transit_idx) + ': ' + str(fit_depth)) #result.params['RpRs'].value**2 * np.cos(impact_parameter_guess)
    #print('fit t0 transit ' + str(transit_idx) + ': ' + str(result.params['t0']))
    #print('fit period transit ' + str(transit_idx) + ': ' + str(result.params['period']))
    #print('fit a transit ' + str(transit_idx) + ': ' + str(result.params['a']))
    #print('fit inc transit ' + str(transit_idx) + ': ' + str(result.params['inc']))
    #print('fit ecc transit ' + str(transit_idx) + ': ' + str(result.params['ecc']))
    #print('fit w transit ' + str(transit_idx) + ': ' + str(result.params['w']))
    #print('fit u1 transit ' + str(transit_idx) + ': ' + str(result.params['u1']))
    #print('fit u2 transit ' + str(transit_idx) + ': ' + str(result.params['u2']))
    
    # plot the results
    if plot_verbosity > 0:
      plt.figure(figsize = [8,4])
      #plt.plot(time[interval], flux[interval], '.', label = 'Flux data')
      #sorted_idxs_interval = np.argsort(time[interval])
      plt.plot(time[interval][sorted_idxs_interval], shifted_flux, '.')#, label = 'Normalized flux')
      #sorted_idxs_mask_interval = np.argsort(masked_time[mask_interval])    
      plt.plot(masked_time[mask_interval][sorted_idxs_mask_interval], shifted_masked_flux, '.', color = 'C0', alpha = 0.25, label = 'Masked')#(quality)')
      #transit_mask_time = transit_masked_time[sorted_idxs_transit_mask_interval]
      #idxs = np.where(np.isin(time, transit_mask_time))
      #plt.plot(transit_mask_time, flux[idxs], '^', color = 'C0', alpha = 0.25, label = 'Masked (sibling transit)')       
      #plt.plot(transit_masked_time[transit_mask_interval], transit_masked_flux[transit_mask_interval], '^', color = 'C0', alpha = 0.25, label = 'Masked (sibling transit)') 
      plt.errorbar(masked_time[mask_interval][sorted_idxs_mask_interval], shifted_masked_flux, shifted_masked_flux_err, color = 'black', ls='none', alpha = 0.25)
      if plot_verbosity > 1:
        #plt.plot(time[interval], flux_data, '.', label = 'Data (after trend removal)')
        plt.plot(time[interval][sorted_idxs_interval], linear_trend, color = 'gray', label = 'Initial baseline trend')
        plt.plot(time[interval][sorted_idxs_interval], new_linear_trend, color = 'purple', label = 'Updated removed baseline trend')
      plt.plot(t, fit_curve_plotting, label = 'Transit fit ')
      plt.plot(t, expected_fit_curve, color = 'black', ls = 'dashed', alpha = 0.5, label = 'Expected transit shape')
      mid = t0_guess + period_guess * transit_idx
      plt.axvline(mid, ls = 'dashed' , color = 'black', alpha = 0.5, label = 'Expected midpoint')
      ax = plt.gca()
      y_min, y_max = ax.get_ylim()
      trans = ax.get_xaxis_transform()#mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
      #ax.fill_between(time[interval], y_min , y_max, where = (time[interval] > mid - duration_guess / 2) * (time[interval] < mid + duration_guess / 2), facecolor='gray', alpha=0.5, transform=trans)
     # ax.fill_between(time[interval], 0.90, y_max , where = (time[interval] > mid - duration_guess / 2) * (time[interval] < mid + duration_guess / 2)  , facecolor='gray', alpha=0.5, transform=trans)
      plt.axvline(mid - duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25, label = 'Expected Ingress/Egress')
      plt.axvline(mid + duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25)
      plt.errorbar(time[interval][sorted_idxs_interval], shifted_flux, shifted_flux_err, color = 'gray', ls='none')#, label = 'Flux uncertainty')
      plt.legend()
      plt.xlabel('Time [Days]')
      plt.ylabel('Relative Flux')
      plt.title(planet_name + ', transit ' + str(int(transit_idx)))
      ax.ticklabel_format(useOffset=False)
      plt.savefig(plots_dir + planet_name + '_t' + str(transit_idx) + '.png')
     
      # if the transit meets the "missing" flag requirements, plot the original flux data in a wider window
      flux_SD_mult  = 3
      #if fit_depth < float(flux_SD_mult * med_flux_SD): # can't impose the second criteria here, as relies on mean depth of individual fits. So we make more expanded plots than there are missing flags in a few (should be rare) cases where a depth is shallower than 3*SD but deeper than 3 sigma below mean.
      make_exp_plots = True
      if make_exp_plots: # actually, let's make the expanded plots for all transits. Will be informative for other flags which we can't check a priori.
          exp_low_bound = t0_guess + period_guess * transit_idx - duration_guess * fit_window_size*2 
          exp_up_bound = t0_guess + period_guess * transit_idx + duration_guess * fit_window_size*2    
          exp_interval = (exp_low_bound < time)*(time < exp_up_bound)   
          mask_exp_interval = (exp_low_bound < masked_time)*(masked_time < exp_up_bound)  
          # evaluate over interpolated time for determing depth and for plotting clarity
          t = np.linspace(time[exp_interval][0], time[exp_interval][-1], 100) # time[interval][sorted_idxs_interval]
          fit_curve_plotting = func(t, result.params['t0'], result.params['period'], result.params['RpRs'], 
                            result.params['a'], result.params['inc'], result.params['ecc'], result.params['w'], 
                            result.params['u1'], result.params['u2'], low_bound, up_bound)
          expected_fit_curve = func(t, midpoint_guess, float(mean_fit_params['period']), float(mean_fit_params['RpRs']), 
                            float(mean_fit_params['a']), float(mean_fit_params['inc']), float(mean_fit_params['ecc']), float(mean_fit_params['w']), 
                            float(mean_fit_params['u1']), float(mean_fit_params['u2']), low_bound, up_bound)
          # redo the plotting, ubut use a 5 times wider window and use the raw data
          plt.figure(figsize = [8,4])
          plt.plot(time[exp_interval], flux[exp_interval]/np.nanmedian(flux[exp_interval]), '.', label = 'Undetrended flux')
          plt.plot(masked_time[mask_exp_interval], masked_flux[mask_exp_interval]/np.nanmedian(flux[exp_interval]), '.', color = 'C0', alpha = 0.25, label = 'Masked')#(quality)')
          plt.plot(time[interval][sorted_idxs_interval], new_linear_trend/np.nanmedian(flux[exp_interval]), color = 'k', ls = 'dotted', label = 'Removed baseline trend')
          plt.errorbar(time[exp_interval], flux[exp_interval]/np.nanmedian(flux[exp_interval]), yerr = flux_err[exp_interval], color = 'black', ls='none', alpha = 0.25)
          plt.plot(t, fit_curve_plotting, label = 'Transit fit ')
          plt.plot(t, expected_fit_curve, color = 'black', ls = 'dashed', alpha = 0.5, label = 'Expected transit shape')
          mid = t0_guess + period_guess * transit_idx
          plt.axvline(mid, ls = 'dashed' , color = 'black', alpha = 0.5, label = 'Expected midpoint')
          ax = plt.gca()
          y_min, y_max = ax.get_ylim()
          x_min, x_max = ax.get_xlim()
          #trans = ax.get_xaxis_transform()#mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
          #plt.axvline(mid - duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25, label = 'Expected Ingress/Egress')
          #plt.axvline(mid + duration_guess / 2, ls = 'dashed' , color = 'black', alpha = 0.25)
          #ax.axvspan(up_bound, x_max, alpha=0.05, color='k', label = 'Not used in fitting') 
          #ax.axvspan(x_min, low_bound, alpha=0.05, color='k')
          ax.axvspan(low_bound, up_bound, alpha=0.1, color='C1', label = 'Fitting window')
          plt.legend()
          plt.xlabel('Time [Days]')
          plt.ylabel('Relative Flux')
          plt.title(planet_name + ', transit ' + str(int(transit_idx)) + ' expanded view')
          ax.ticklabel_format(useOffset=False)
          plt.xlim([x_min,x_max])
          plt.savefig(plots_dir + planet_name + '_t' + str(transit_idx) + '_zoomout.png')       

    return result.params['t0'].value, result.params['t0'].stderr, result.params['RpRs'].value, result.params['RpRs'].stderr, fit_depth, chi_sq, BIC, 0

  t0_arr = []
  depth_arr = []
  t0_err_arr = []
  depth_err_arr = []
  BIC_arr = []
  skipped = []
  skipped_bad_fit = np.zeros(num_transits)
  skipped_missing_data = np.zeros(num_transits)
  no_points = [] # which transits have no data in their fit window
    
  for transit_idx in np.linspace(0,num_transits-1,num_transits):
    t0, t0_stderr, RpRs, RpRs_stderr, fit_depth, chi_sq, BIC, skip = make_fit(transit_idx, no_points, 
                                                                                plot_verbosity = 1, verbosity = 0, 
                                                                                plots_dir = plots_dir, 
                                                                                flag_plot_path = flag_plot_path)
    print(' fitting transit: ' + str(transit_idx))
    try:
      depth_err_arr += [2*(RpRs**2)*RpRs_stderr / RpRs]
      t0_err_arr += [t0_stderr]
      t0_arr += [t0]
      depth_arr += [fit_depth]  
    except TypeError: 
      depth_err_arr += [np.nan]
      t0_err_arr += [np.nan]
      depth_arr += [np.nan]
      t0_arr += [np.nan]
      skipped += [transit_idx.astype(int)]
      skipped_bad_fit[transit_idx.astype(int)] = 1
      print('errors are of type None; fit likely did not converge properly.')
    BIC_arr += [BIC]
    if skip:
      skipped += [transit_idx.astype(int)]
      skipped_missing_data[transit_idx.astype(int)] = 1

  depth_arr = np.array(depth_arr)
  depth_err_arr = np.array(depth_err_arr)
  t0_arr = np.array(t0_arr)
  t0_err_arr = np.array(t0_err_arr)
  BIC_arr = np.array(BIC_arr)

  t0_arr[skipped_missing_data.astype(bool)] = [-1]
  t0_arr[skipped_bad_fit.astype(bool)] = [-2]  

  # visualize which were skipped due to missing data
  which_skipped = np.zeros(num_transits); which_skipped[skipped] = 1
  plot_skipped = 1
  if plot_skipped:
    plt.figure(figsize = [15,2])
    plt.plot(np.zeros(num_transits),'o')
    plt.plot(skipped,np.zeros(len(skipped)),'o', label = 'skipped')
    plt.xlabel('Transit Index'); plt.legend(); plt.yticks(ticks = [])
    plt.title(planet_name)
    if shift_value > 0:
      print('TESTING! This spectrum was shifted.')
    print(str(len(skipped)) + ' skipped out of ' + str(num_transits) + ' total transits.')

  # calculate deltas
  t0_arr_clean = np.copy(t0_arr)
  t0_arr_clean[t0_arr_clean < 0] = np.nan
  delta_midpoints = np.array(expected_midpoints - t0_arr_clean)
  delta_depths = np.array(expected_depth - depth_arr)


  if shift_value == 0:
  # determine overall variability in transit timings and depths
    sd_ttv = np.nanstd(delta_midpoints) 
    max_ttv = np.nanmax(np.abs(delta_midpoints)) 
    mean_ttv = np.nanmedian(delta_midpoints) 
    mean_depth = np.nanmedian(depth_arr)
    sd_delta_depth = np.nanstd(delta_depths)
    max_delta_depth = np.nanmax(np.abs(delta_depths))
    mean_delta_depth = np.nanmedian(delta_depths)
    sd_depth = np.nanstd(depth_arr - mean_depth)
    if verbosity > 0:
      print('TTV\'s: max abs tt variation ' + "{:.2e}".format(max_ttv) + ' [min], standard deviation ' + "{:.2e}".format(sd_ttv * 1440) + ' [min], mean tt variation ' + "{:.2e}".format(mean_ttv) )
      print('Depths: max variation ' + "{:.2e}".format(max_delta_depth) + ', standard deviation ' + "{:.2e}".format(sd_delta_depth))     
        
  # assign sigma thresholds for flags
  depth_threshold = 3          # number of standard deviations 
  ttv_threshold = 5            # number of standard deviations 
  flux_SD_mult = 3             # NOTE: if chnage this here, also change on line 612
  #width_threshold = 3          # number standard deviations is width away from expected width (or mean width)

  # assign a set of candidate flags 
#  # depth is too great  
#  depth_flag = (which_skipped == 0) * ((delta_depths > (mean_delta_depth + depth_threshold*sd_delta_depth)) + (delta_depths < (mean_delta_depth - depth_threshold*sd_delta_depth)))
#  print('here4')
#      # Note: if delta_depths = 0, expected = fit
#      #       if delta_depths > 0, fit depth is larger than expected and negative 
#      #       if 0 > delta_depths > expected depth, fit depth is negative and smaller in magnitude than expected
#      #       if delta_depths < expected_deoth, fit depth is positve
#  # transit is actually a peak in flux
#  pos_depth_flag = (which_skipped == 0) * (depth_arr < 0) # depth is > 0 for a dip in flux, < 0 for a spike in flux
#  # transit is consistent with being flat (ie. no transit)
#  # flat_curve_flag =  (which_skipped == 0) * (np.abs(depth_arr) < mean_flux_err) #flux_SD)  
#  # TESTING THIS!!
#  flat_curve_flag = (which_skipped == 0) * (np.abs(depth_arr) < 2 * mean_flux_SD) 
#  flatness = mean_flux_err / np.abs(depth_arr) # parameter to store the degree of "flatness" relative to flux error
#  #flat_curve_flag_2 = (skipped == 0) * (chi_sq_arr_p <= chi_sq_arr_g) # possibly should re-define a flat fit and use this flag
#  # large transit timing variation
#  tt_flag = (which_skipped == 0) * ((delta_midpoints > mean_ttv + ttv_threshold*sd_ttv) + (delta_midpoints < mean_ttv - ttv_threshold*sd_ttv))

 # NEW FLAGS (5/24/22)
  flat_curve_flag = (which_skipped == 0) * (np.abs(depth_arr) < float(flux_SD_mult * med_flux_SD)) * (np.abs(depth_arr - mean_depth) > depth_threshold * sd_depth)  
  pos_depth_flag = (which_skipped == 0) * (depth_arr < 0) # depth is > 0 for a dip in flux, < 0 for a spike in flux (didn't change this))
  depth_flag = (which_skipped == 0) * (depth_arr > mean_depth + depth_threshold * sd_depth)
  tt_flag = (which_skipped == 0) * (np.abs(delta_midpoints - mean_ttv) >  ttv_threshold*sd_ttv) # same as before, just cleaner notation
  flatness = float(flux_SD_mult * med_flux_SD) / np.abs(depth_arr) # parameter to store the degree of "flatness" relative to flux error
 
  not_skipped = (which_skipped == 0) 
  #dur_flag = (widths > mean_width + width_threshold*sd_width) + (widths < mean_width - width_threshold*sd_width)  
  if verbosity > 0:
    print(str(np.round(len(np.where(depth_flag)[0]))) + ' transits meet depth flag:')
    print('      ' + str(np.where(depth_flag)))
    print(str(np.round(len(np.where(pos_depth_flag)[0]))) + ' transits meet postive depth flag:')
    print('      ' + str(np.where(pos_depth_flag)))
    print(str(np.round(len(np.where(tt_flag)[0]))) + ' transits meet transit timing flag')
    print('      ' + str(np.where(tt_flag)))
    print(str(np.round(len(np.where(flat_curve_flag)[0]))) + ' transits meet flat transit 1 flag')
    print('      ' + str(np.where(flat_curve_flag)))
    
  # visualize what is flagged
  if plot_flags:
    fig = plt.figure(figsize = [10,8])
    plt.subplot(211)
#    plt.plot(expected_midpoints[not_skipped], delta_depths[not_skipped], '.')
#    plt.axhline(mean_delta_depth, label = 'mean')
#    plt.axhline(mean_delta_depth + depth_threshold*sd_delta_depth,  label = str(depth_threshold) + '*SD above mean', color = 'gray')
#    plt.axhline(mean_delta_depth - depth_threshold*sd_delta_depth,  label = str(depth_threshold) + '*SD below mean', color = 'gray')
#   plt.plot(expected_midpoints[depth_flag], delta_depths[depth_flag], '.', label = 'Depth flagged')
#    #plt.plot(expected_midpoints[skipped], delta_depths[skipped], 'o', color = 'gray', label = 'Skipped (missing data)')
#    #plt.errorbar(expected_midpoints[not_skipped], delta_depths[not_skipped], yerr = depth_err_arr[not_skipped], ls='none', label = 'standard error')
#    plt.ylabel('Depth deltas')
#    plt.xlabel('Expected transit time [Days]')
#    #plt.ylim([-0.001, 0.001])
#    #plt.title('Guassian + polynomial fitting')
#    plt.legend()
    plt.plot(expected_midpoints, delta_midpoints * 1440, '.', color = 'k')
    plt.ylabel('ttv [Min]')
    plt.xlabel('Expected transit time [Days]')
    plt.axhline(mean_ttv * 1440, ls = 'dashed', alpha = 0.5, color = 'k', label = 'mean')
    plt.axhline(mean_ttv * 1440 + ttv_threshold*sd_ttv * 1440,  label = str(ttv_threshold) + '*SD above mean', ls = 'dashed', color = 'C0', alpha = 0.5)
    plt.axhline(mean_ttv * 1440 - ttv_threshold*sd_ttv * 1440,  label = str(ttv_threshold) +'*SD below mean', ls = 'dashed', color = 'C0', alpha = 0.5)
    plt.plot(expected_midpoints[tt_flag], delta_midpoints[tt_flag] * 1440, '.', color = 'C0', label = 'Flagged for large TTV')
    #ttv_range = np.abs(np.nanmax(delta_midpoints * 1440) - np.nanmin(delta_midpoints * 1440)) * 1.5
    ylims = plt.gca().get_ylim()
    plt.errorbar(expected_midpoints[np.invert(tt_flag)], delta_midpoints[np.invert(tt_flag)] * 1440, yerr = np.array(t0_err_arr[np.invert(tt_flag)]) * 1440, ls='none', color = 'k', alpha = 0.5)
    plt.errorbar(expected_midpoints[tt_flag], delta_midpoints[tt_flag] * 1440, yerr = np.array(t0_err_arr[tt_flag]) * 1440, ls='none', color = 'C0', alpha = 0.5)
    plt.ylim([ylims[0], ylims[1]])#[(mean_ttv - ttv_range/2), (mean_ttv + ttv_range/2)])
    #plt.title('Guassian + polynomial fitting')
    plt.legend()

    #plt.figure(figsize = [15,4])
    plt.subplot(212)
    plt.plot(expected_midpoints[not_skipped], depth_arr[not_skipped], '.', color = 'k')
    plt.axhline(mean_depth, ls = 'dashed', label = 'Mean depth', color = 'k', alpha = 0.5)
    #plt.axhline(float(mean_fit_depth), label = 'Expected depth', color = 'k', alpha = 0.5)    
    plt.axhline(mean_depth + depth_threshold*sd_depth,  label = str(depth_threshold) + '*SD above mean', ls = 'dashed', color = 'C0', alpha = 0.5)
    plt.plot(expected_midpoints[depth_flag], depth_arr[depth_flag], '.', label = 'Flagged for large depth')
    plt.axhline(float(flux_SD_mult * med_flux_SD), label = str(flux_SD_mult) + ' * Mean flux SD', ls = 'dashed', color = 'C1', alpha = 0.5)
    plt.plot(expected_midpoints[flat_curve_flag], depth_arr[flat_curve_flag], '.', label = 'Flagged as missing')
    plt.plot(expected_midpoints[pos_depth_flag], depth_arr[pos_depth_flag], '.', label = 'Flagged for positive depth')
    plt.errorbar(expected_midpoints[pos_depth_flag], depth_arr[pos_depth_flag], yerr = depth_err_arr[pos_depth_flag], ls='none', color = 'C2', alpha = 0.5)
    #plt.plot(expected_midpoints[skipped], delta_depths[skipped], 'o', color = 'gray', label = 'Skipped (missing data)')
    #plt.errorbar(expected_midpoints[not_skipped], delta_depths[not_skipped], yerr = depth_err_arr[not_skipped], ls='none', label = 'standard error')  
    #depth_range = np.abs(np.nanmax(depth_arr) - np.nanmin(depth_arr)) * 1.5
    ylims = plt.gca().get_ylim()
    not_flagged = np.invert(depth_flag)*np.invert(flat_curve_flag)
    plt.errorbar(expected_midpoints[not_skipped*not_flagged], depth_arr[not_skipped*not_flagged], yerr = depth_err_arr[not_skipped*not_flagged], ls='none', color = 'k', alpha = 0.5)
    plt.errorbar(expected_midpoints[depth_flag], depth_arr[depth_flag], yerr = depth_err_arr[depth_flag], ls='none', color = 'C0', alpha = 0.5) 
    plt.errorbar(expected_midpoints[flat_curve_flag], depth_arr[flat_curve_flag], yerr = depth_err_arr[flat_curve_flag], ls='none', color = 'C1', alpha = 0.5)
    plt.ylim([ylims[0], ylims[1]])#plt.ylim([(mean_depth - depth_range/2), (mean_depth + depth_range/2)])
    plt.ylabel('Fit Transit Depth [Relative Units]')
    plt.xlabel('Expected transit time [Days]')
    #plt.ylim([-0.001, 0.001])
    #plt.title('Guassian + polynomial fitting')
    plt.legend()
    
    #print('here4')
    #plt.figure(figsize = [15,4])
    #plt.subplot(413)
    #plt.plot(expected_midpoints, depth_arr, '.')
    #plt.plot(expected_midpoints[pos_depth_flag], depth_arr[pos_depth_flag], '.', label = 'Pos transit depth')
    #plt.axhline(float(mean_fit_depth), label = 'Expected depth', color = 'gray')
    #plt.errorbar(expected_midpoints, depth_arr, yerr = depth_err_arr, ls='none', label = 'standard error')
    #plt.ylabel('Fit Depths')
    #plt.xlabel('Expected transit time [Days]')
    #plt.title('Guassian + polynomial fitting')
    #plt.ylim([-0.001, 0.001])
    #plt.legend()
    #print('here5')
    #plt.figure(figsize = [15,4])
    #plt.subplot(414)
    #plt.plot(expected_midpoints, depth_arr, '.')
    #plt.plot(expected_midpoints[flat_curve_flag], depth_arr[flat_curve_flag], '.', label = 'flat curve')
    #plt.errorbar(expected_midpoints, depth_arr, yerr = depth_err_arr, ls='none', label = 'standard error')
    #plt.axhline(mean_flux_err, label = 'mean error in lightcurve flux')
    #plt.axhline(expected_depth, label = 'Expected depth', color = 'gray')
    #plt.ylabel('Fit Depths')
    #plt.xlabel('Expected transit time [Days]')
    #plt.title('Guassian + polynomial fitting')
    #plt.ylim([-0.001, 0.001])
    #plt.legend()
    #print('here6')
    fig.savefig(flag_plot_path)

  return  depth_arr, t0_arr, depth_err_arr, t0_err_arr, BIC_arr, depth_flag, pos_depth_flag, flat_curve_flag, tt_flag, skipped, skipped_missing_data, skipped_bad_fit, expected_params, mean_fit_params, time, flux, flatness#, crowdsap



# ----------- Run the fitting ------------ # 

planets_to_run = open(in_file, "r").read().split(', ')
planets_to_run = planets_to_run[start:stop]

#planets_to_run = ['kepler-87b']
#planets_to_run = ['kepler-138c']
#planets_to_run = ['kepler-101b', 'kepler-101c', 'kepler-11b', 'kepler-11c', 'kepler-11d', 'kepler-11e', 'kepler-11f', 'kepler-11g',
#                  'kepler-117b', 'kepler-117c','kepler-138b', 'kepler-138c', 'kepler-138d', 'kepler-30b', 'kepler-30c', 'kepler-30d', 
#                  'kepler-462b', 'kepler-462c', 'kepler-60b', 'kepler-60c', 'kepler-60d', 'kepler-87b', 'kepler-87c']
#planets_to_run = planets_to_run[12:]

ran = []
failed = []
messages = []

#planet_name = planets_to_run_for_others[1]
#depth_arr, depth_err_arr, t0_err_arr, BIC_arr, depth_flag, pos_depth_flag, flat_curve_flag, tt_flag = make_fits(planet_name, shift_value = 0, plot_lightcurve = False, verbosity = 0, plot_verbosity = 0, plot_flags = 1)

for i in range(len(planets_to_run)):
 # try:
  #planet_name = 'Kepler-1655 b' # CHANGE THIS BACK!!
  planet_name = planets_to_run[i]

    
  dt = datetime.datetime.now()
  timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X") 
    
  #if os.path.isdir(out_dir):
  #  os.rename(out_dir, out_dir + '_' + timestamp)
  #os.mkdir(out_dir)
    
  planet_dir = out_dir + '/' + planet_name
  print('planet_name:' + str(planet_name))
  if os.path.isdir(planet_dir):
        os.rename(planet_dir, planet_dir + '_' + timestamp)

  os.mkdir(planet_dir)
  results_file = planet_dir + '/Fit_data_' + planet_name + '.csv'
  plots_dir = planet_dir + '/transit_plots/'
  os.mkdir(plots_dir)
  flag_plot_path = planet_dir + '/flags_plot_' + planet_name + '.png'

  try:
    depth_arr, t0_arr, depth_err_arr, t0_err_arr, BIC_arr, depth_flag, pos_depth_flag, flat_curve_flag, tt_flag, skipped, skipped_missing_data, skipped_bad_fit, expected_params, mean_fit_params, time, flux, flatness = make_fits(planet_name, shift_value = 0,  plots_dir = plots_dir, flag_plot_path = flag_plot_path, plot_lightcurve = True, verbosity = 0, plot_verbosity = 0, plot_skipped = 0, plot_flags = 1)

    ran += [planet_name]
    df = pd.DataFrame(t0_arr, columns = ['Transit_midpoint [Days]'])   
    #df.to_csv('./transit_times_'+ planet_name + '.csv') # for Sofia Sheikh save this separatley 

    # save the results
    #data = pd.DataFrame([depth_arr, depth_err_arr, t0_arr, t0_err_arr, BIC_arr, depth_flag, pos_depth_flag, flat_curve_flag, tt_flag, 
    #                      skipped_missing_data, skipped_bad_fit,],
    #                  index = ['Fit depths', 'Fit depths error', 'Fit transit times', 'Fit transit times error', 'BIC', 'Large depth flag', 
    #                            'Positve depth flag', 'Flat curve flag', 'Large TTV flag', 'Skipped (missing data)', 'Skipped (bad fit)'])

    save_data  = np.transpose(np.vstack([depth_arr, depth_err_arr, t0_arr, t0_err_arr, BIC_arr, depth_flag, pos_depth_flag, flat_curve_flag, tt_flag, flatness, skipped_missing_data, skipped_bad_fit]))
    print('here10')
    # save the results
    data = pd.DataFrame(save_data,
                      columns = ['Fit_depths', 'Fit_depths_error', 'Fit_transit_times', 'Fit_transit_times_error', 'BIC', 'Large_depth_flag', 'Positive_depth_flag', 'Flat_curve_flag', 'Large_TTV_flag', 'Flatness_parameter', 'Skipped (missing data)', 'Skipped (bad fit)'])


    data.to_csv(results_file)   
    # convert the mean fit params to a dictionary as well
    mean_fit_params_dict = mean_fit_params.to_dict('records')[0]
    mean_fit_params_dict.pop('Name')
    with open(results_file, 'a+') as csvfile:
      csvwriter = csv.writer(csvfile) 
      csvwriter.writerow('') 
      csvwriter.writerow(['Expected (NASA exoplanet archive) parameter values:'])   
      csvwriter.writerow(expected_params.keys()) 
      csvwriter.writerow(expected_params.values()) 
      csvwriter.writerow(['Mean fit parameter values:'])   
      csvwriter.writerow(mean_fit_params_dict.keys()) 
      csvwriter.writerow(mean_fit_params_dict.values())                                                                  
      csvwriter.writerow('')
      csvwriter.writerow(['Note on skipped transits:'])
      csvwriter.writerow(['For transit times:  -1 --> skipped transit due to missing data'])
      csvwriter.writerow(['                    -2 --> skipped transit due to bad fitting'])
      csvwriter.writerow(['For all other parameters:  Nan --> skipped for any reason'])

    print('here11')    
    
  except Expected_Depth_Exception: # the expected depth was less than twice the ligthcurve mean flux error so no fitting was attempted
    print('here12')
    file = open(planet_dir + '/Log.txt','w')
    file.write('The expected depth was less than twice the ligthcurve mean flux error so no fitting was attempted.')
    file.write('Depth within lightcurve uncertainty means any fit depths would not be meaningful.')
    fail_type, value, traceback = sys.exc_info()
    line_num = traceback.tb_lineno
    fail_message = str(fail_type).split('\'')[1] + ': ' + str(value)
    print('Failed to run for ' + planet_name + ': ' + fail_message + ' (error on line ' + str(line_num) + ')') 
    failed += [planet_name]
    messages += [fail_message]
    
  except: # any other failure
    print('here13')
    file = open(planet_dir + '/Log.txt','w')
    file.write('An unkown error occured and this planet failed to run.')
    fail_type, value, traceback = sys.exc_info()
    line_num = traceback.tb_lineno
    fail_message = str(fail_type).split('\'')[1] + ': ' + str(value)
    print('Failed to run for ' + planet_name + ': ' + fail_message + ' (error on line ' + str(line_num) + ')') 
    tb.print_exc()
    failed += [planet_name]
    messages += [fail_message]    
    

  #except Exception:
  #  fail_type, value, traceback = sys.exc_info()
  #  fail_message = str(fail_type).split('\'')[1] + ': ' + str(value)
  #  print('Failed to run for ' + planet_name + ': ' + fail_message)
  #  failed += [planet_name]
  #  messages += [fail_message]


print('')
print('Ran: ')
print(ran)
print('Failed:  ')
print(failed)
print('Messages:  ')
print(messages)

