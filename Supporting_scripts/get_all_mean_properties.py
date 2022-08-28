# similar to get_all_stacked_properties.py, but instead of folding into phase space, fit the entire lightcurve in time units

import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
NASA_exoplanet_table = pd.read_csv('NASA_exoplanet_archive.csv', skiprows = 292)
from lmfit import Model
from lmfit import Parameter
from astroquery.simbad import Simbad 
import batman
import astropy.units as u
import sys
import datetime
import os
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Run stacked fitting.")
parser.add_argument("-start", "--start", type=int, help="Index of planet to start on.")
parser.add_argument("-stop", "--stop", type=int, help="Index of planet to stop on.")
parser.add_argument("-outfile", "--outfile", type=str, help="output file.")
args = parser.parse_args()
start = args.start
stop = args.stop
outfile = args.outfile

#KOIs = pd.read_csv('./exoplanet_archive_KOIs.csv')
#confirmed = KOIs[KOIs['koi_disposition'] == 'CONFIRMED']
kep_names = open('planet_list_236_updated.txt', "r").read().split(', ') #confirmed['kepler_name'].tolist()
kep_names = kep_names

# make a new dataframe to store ALL fit parameters for each stacked fit
data = pd.read_csv('Mean_flux_errs.csv')
#data['EXPECTED_DEPTH'] = old_expected_depth
#data['MEAN_FIT_DEPTH'] = np.nan
#data['DEPTH_ERR_RATIO'] = np.nan

all_params = pd.DataFrame(columns = ['Name', 'RpRs', 'period', 'a', 'inc', 'ecc', 'w', 'u1', 'u2', 'depth', 'chi square'],
                         index = data.index)
all_params['Name'] = data['NAME']


# Linear trend
def linear_func(x, a, b):
      return a*x + b

# Do my own simplistic stiching in several ways
def stitch(lc_collection): 

    tot_time = np.zeros(0) # to store concatenated arrays
    tot_flux = np.zeros(0)
    tot_flux_err = np.zeros(0)
    for i in range(len(lc_collection)):
        lc = lc_collection[i]
        flux = lc.flux.value
        time = lc.time.value
        flux_err = lc.flux_err.value
        rel_flux_err = flux_err/flux
        nan_mask = np.invert(np.isnan(flux))
        
        # Fit and remove linear trend
        popt, pcov = curve_fit(linear_func, time[nan_mask], flux[nan_mask])
        linear_trend = linear_func(time, *popt) # evaluate over the whole interval
        norm2 = flux / linear_trend
        
        tot_time = np.hstack([tot_time, time])
        tot_flux = np.hstack([tot_flux, norm2])
        tot_flux_err = np.hstack([tot_flux_err, rel_flux_err])
        
    return tot_time, tot_flux, tot_flux_err

# define Batman fit function
def func(x, t0, period, RpRs, a, inc, ecc, w, u1, u2):#, low_bound, up_bound):

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
    #interval = (low_bound < x)*(x < up_bound) # interval to fit on
    t = x#[interval]                              # times to calculate at
    #print(t)
    #print(x)
    
    m = batman.TransitModel(params, t)              # initializes model
    model_flux = m.light_curve(params)              # calculates light curve
    return model_flux

# make the fits
failed = []

#start = 0 # 710 # ONLY becuase it got disconnected at this star
#stop = len(kep_names)

for i in np.arange(start,stop,1):
    try: 
        planet_name  =  kep_names[i] #'Kepler-37 d'#'Kepler-230 b'
        query_name = planet_name
        if planet_name == 'Kepler-1 b':  query_name = 'TrES-2 b'
        if planet_name == 'Kepler-458 b': query_name = 'KIC 9663113 b'
        if planet_name == 'Kepler-324 d': query_name = 'KOI-1831 d'    
        if planet_name == 'Kepler-1703 b':  query_name = 'KOI-3503 b'
        if planet_name == 'Kepler-968 d':  query_name = 'KOI-1833 d'
        if planet_name == 'Kepler-460 b':  query_name = 'KIC 5437945 b'
        if planet_name == 'Kepler-86 b':  query_name = 'PH2 b'
        if planet_name == 'Kepler-1703 c':  query_name = 'KOI-3503 c' 

        print(planet_name)

        planet =  NASA_exoplanet_table[NASA_exoplanet_table['pl_name'] == query_name]
        if planet.empty:
            print('trying alternate names.')
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

        period_expected = np.nanmedian(planet['pl_orbper']) # Planet period [Days]
        t0_expected = np.nanmedian(planet['pl_tranmid']) - 2454833 # Transit midpoint [Days]
        R_p = np.nanmedian(planet['pl_rade']) * 0.0091577 # Planet radius [Ro]
        R_star = np.nanmedian(planet['st_rad'])  # Star radius [Ro]
        R_star_AU = R_star * 0.00465047
        impact_parameter_expected = np.nanmedian(planet['pl_imppar']) # impact parameter 
        duration_expected = np.nanmedian(planet['pl_trandur']) / 24 # Planet transit duration [Days]
        semi_maj_ax_expected = np.nanmedian(planet['pl_orbsmax']) / R_star_AU # semi-major axis [Stellar Radii]
        inc_expected = np.nanmedian(planet['pl_orbincl']) # inclination [deg]
        arg_of_periastron_expected = np.nanmedian(planet['pl_orblper']) # arguement/longitude of periastron/periapse [deg]
        if np.isnan(arg_of_periastron_expected):
            arg_of_periastron_expected = 0 # Possibly this is an ok default 
        else:
            arg_of_periastron_expected = arg_of_periastron_expected
        ecc_expected =  np.nanmedian(planet['pl_orbeccen']) # eccentricity
        RpRs_expected = R_p/R_star # Planet star radius ratio ~= sqrt(transit depth)
        u_expected = [0.5, 0.1, 0.1, -0.1] # Possibly this is an ok default    
        u1_expected = u_expected[0]; u2_expected = u_expected[1]
        depth_expected = np.abs((RpRs_expected**2) * np.cos(impact_parameter_expected))

        # UPDATE THIS TO USE KIC NAME
        search_result = lk.search_lightcurve(planet_name, author='kepler', cadence='long') 
        lc_collection = search_result.download_all()
        lc = lc_collection.stitch()#.remove_outliers()
        time, flux, tot_flux_err = stitch(lc_collection)
        #tot_time, tot_flux, tot_flux_err = stitch(lc_collection)
        #period = np.linspace(1, 20, 10000)#np.linspace(0.9 *expected_period, 1.1 * expected_period, 10000) # np.linspace(1, 20, 10000)
        # Create a BLSPeriodogram
        #bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);
        #planet_period = bls.period_at_max_power 
        #planet_t0 = bls.transit_time_at_max_power
        #planet_dur = bls.duration_at_max_power
        #lc_folded = lc.fold(period=period_expected, epoch_time=t0_expected)
        #folded_time = lc_folded.time.value
        #folded_flux = lc_folded.flux.value
        #folded_flux_err = lc_folded.flux_err.value
        #ax = lc_folded.scatter()
        #ax.set_ylim([0.998, 1.0025])

        #ax = lc.fold(period=planet_period, epoch_time=planet_t0).scatter()
        #bls.show_properties()    

        # create model of folded curve
        fmodel = Model(func)
        

        # create a mask to only feed the model the regions around transits (so it doesn't fit other planets' transits)
        num_transits = int(np.floor((time[-1] - t0_expected) / period_expected)) + 1
        expected_midpoints = np.linspace(t0_expected, t0_expected + (num_transits - 1) * period_expected, num_transits)
        transit_mask = np.zeros(len(time)).astype(bool)
        tol = duration_expected * 3
        for t in expected_midpoints:
            interval = np.where((time < t + tol) * (time > t - tol))
            transit_mask[interval] = 1
            
        #low_bound = 0 - 3 * duration_expected
        #up_bound = 0 + 3 * duration_expected
        #interval = (folded_time >= low_bound) * (folded_time <= up_bound)
        nan_mask = np.invert(np.isnan(flux))
        mask = nan_mask * transit_mask
        
        # make fit
        #result = fmodel.fit(data=flux[mask], x=time[mask],    #     0.8                          1.2
        #  period = Parameter('period', value = period_expected, vary = True, min = 0.5 * period_expected, max = 2 * period_expected, brute_step = 0.1 * period_expected), #False),
         # a = Parameter('a', value = semi_maj_ax_expected, vary = False),
         # inc = Parameter('inc', value = 90, vary = True, min = 80, max = 100),#inc_expected, vary = False),# True, min = 0, max = 180),
         # ecc = Parameter('ecc', value = ecc_expected, vary = False),
         # w = Parameter('w', value = arg_of_periastron_expected, vary = False),
         # u1 = Parameter('u1', value = u1_expected, vary = False),
         # u2 = Parameter('u2', value = u2_expected, vary = False),
         # RpRs = Parameter('RpRs', value = RpRs_expected, vary = True, min = 0.8 * RpRs_expected, max = 10 * RpRs_expected, brute_step = 0.1 * RpRs_expected),
         # t0 = Parameter('t0', value = t0_expected, vary = False),# min = low_bound, max = up_bound), #folded_time[int(len(folded_time)/2)]
         # #low_bound = Parameter('low_bound', value = low_bound, vary = False),
         # #up_bound = Parameter('up_bound', value = up_bound, vary = False),
         # method = 'brute',
         # calc_covar = True)    
        
        
        # experimental -- holding inc const        
        result = fmodel.fit(data=flux[mask], x=time[mask],    #     0.8                          1.2
          period = Parameter('period', value = period_expected, vary = True, min = 0.5 * period_expected, max = 2 * period_expected, brute_step = 0.1 * period_expected), #False),
          a = Parameter('a', value = semi_maj_ax_expected, vary = False),
          inc = Parameter('inc', value = 90, vary = False),#, min = 80, max = 100),#inc_expected, vary = False),# True, min = 0, max = 180),
          ecc = Parameter('ecc', value = ecc_expected, vary = False),
          w = Parameter('w', value = arg_of_periastron_expected, vary = False),
          u1 = Parameter('u1', value = u1_expected, vary = False),
          u2 = Parameter('u2', value = u2_expected, vary = False),
          RpRs = Parameter('RpRs', value = RpRs_expected, vary = True, min = 0.8 * RpRs_expected, max = 10 * RpRs_expected, brute_step = 0.1 * RpRs_expected),
          t0 = Parameter('t0', value = t0_expected, vary = False),# min = low_bound, max = up_bound), #folded_time[int(len(folded_time)/2)]
          #low_bound = Parameter('low_bound', value = low_bound, vary = False),
          #up_bound = Parameter('up_bound', value = up_bound, vary = False),
          method = 'brute',
          calc_covar = True) 

        red_chi_sq = result.redchi
        
        fit_curve = func(time[mask], result.params['t0'], result.params['period'], result.params['RpRs'], 
                        result.params['a'], result.params['inc'], result.params['ecc'], result.params['w'], 
                        result.params['u1'], result.params['u2'])#, low_bound, up_bound)

        #bls_model = bls.get_transit_model(period = period_expected * u.d, transit_time = t0_expected * u.d, duration = duration_expected * u.d).fold(period_expected * u.d,
        #                                   t0_expected * u.d)

        #fit_depth = 1 - np.nanmin(bls_model.flux.value)
        fit_depth = 1 - np.nanmin(fit_curve)

        make_plots = False
        if make_plots:
            plt.figure(figsize = [15,5])
            plt.plot(time[mask], flux[mask], '.', alpha = 0.6)
            plt.plot(time[mask], fit_curve, label = 'Batman model')
            #plt.plot(bls_model.time.value, bls_model.flux.value, label = 'BLS model')
            ymin, ymax = plt.gca().get_ylim()
            plt.ylim([ymin, 1.005])
            plt.xlim(low_bound, up_bound)
            plt.xlabel('Time [Days]'); plt.ylabel('Normalized Flux')
            plt.plot([],[], ls = None, label = 'Expected depth: ' + str(np.round(depth_expected,5)) + '\n' + 'Fit depth: ' + str(np.round(fit_depth,5)))
            plt.title(planet_name)
            plt.legend()

        # save depth/error ratio information    
        idx = np.where(data['NAME'] == planet_name)[0][0]
        #print('shape data df: ' + str(data.shape))
        #print('index of planet in data df: ' + str(idx))
        #mean_err = data['MEAN_FLUX_ERR'][idx]
        #col = 3 # fit depth
        #data.iloc[idx,col] = fit_depth
        #col = 4 # depth/err ratio
        #data.iloc[idx,col] = fit_depth / mean_err
        
        #print(result.params)
        
        # save fit parameters
        all_params.iloc[idx,1] = result.params['RpRs'].value
        all_params.iloc[idx,2] = result.params['period'].value
        all_params.iloc[idx,3] = result.params['a'].value
        all_params.iloc[idx,4] = result.params['inc'].value
        all_params.iloc[idx,5] = result.params['ecc'].value 
        all_params.iloc[idx,6] = result.params['w'].value
        all_params.iloc[idx,7] = result.params['u1'].value
        all_params.iloc[idx,8] = result.params['u2'].value
        all_params.iloc[idx,9] = fit_depth
        all_params.iloc[idx,10] = result.redchi
        
        
    except:
        fail_type, value, traceback = sys.exc_info()
        line_num = traceback.tb_lineno
        print('Failed: ' + planet_name)
        fail_msg = str(fail_type).split('\'')[1] + ': ' + str(value) + ', on line ' + str(line_num)
        print(fail_msg)
        failed += [planet_name + ' (' + fail_msg + ') on line' + str(line_num)]
        print()

        
dt = datetime.datetime.now()
timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X")
if os.path.exists(outfile):
    os.rename(outfile, '_' + timestamp + outfile)
all_params.to_csv(outfile, index = False)
                                                                