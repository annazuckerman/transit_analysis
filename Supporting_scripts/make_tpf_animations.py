import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
#from jinja2.utils import markupsafe
#markupsafe.Markup()
from bokeh.io import output_notebook
import PIL
from PIL import Image
import os
import imageio
import sys
from matplotlib import colors
from matplotlib import patches
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description = "Make  GIF animation of the TPFs.")
#parser.add_argument("-in_file","--input_file", type=str, help= "Filename for text file in this directory containing list of planet names to run on.") 
parser.add_argument("-start", "--start", type=int, help="Index of planet to start on.")
parser.add_argument("-stop", "--stop", type=int, help="Index of planet to stop on.")
parser.add_argument("-norm_tf",  help="Normalize by time-median in each pixel", action = 'store_true') # default is false
parser.add_argument("-log_tf",  help="Plot in log scale", action = 'store_true') # default is false
#parser.add_argument("-out_dir","--output_directory", type=str, help= "output directory.") 
#parser.add_argument("-run_date", "--run_date", type=str,  help= "Date index of run.")
args = parser.parse_args()
start = args.start
stop = args.stop
norm_tf = args.norm_tf 
log_tf = args.log_tf 
#in_file = args.input_file
#out_dir = args.output_directory
#run_date = args.run_date
# NASA exoplanet archive table
NASA_exoplanet_table = pd.read_csv('NASA_exoplanet_archive.csv', skiprows = 292)


# can only have one plotting type at a time for now
if norm_tf and log_tf: 
    raise Exception('Can only have one plotting type at a time for now (can pass only one of norm_tf or log_tf')
    


# IT WILL OVERWRITE!
names =  ['Kepler-51 b', 'Kepler-422 b', 'Kepler-41 b', 'Kepler-433 b', 'Kepler-685 b', 'Kepler-74 b', 'Kepler-548 b'] #open(infile, "r").read().split(', ')
transits = [30, 99, 616, 248, 791, 56, 132]
save = True # to save the gif 
gif_dir = 'TPF_animations' 
results_dir = 'Batman_outputs_228_6-19-22'


# plot with more control (though labeling axes will be a pain)
def plot_tpf(tpf, i, c_min, c_max, planet, transit, t, norm_tf = False, log_tf = False):
    fs = 14
    fig = plt.figure(figsize = [6,6])
    y = tpf[int(indxs[i])].hdu[1].data['FLUX'][0,:,:]
    if norm_tf:
        y = y - pixel_median + 1
        norm_c_max = c_max - np.nanmax(pixel_median) + 1 # assumes the overall max is in the same pixel as the max of the median of the pixels
        norm_c_min = c_min - np.nanmin(pixel_median) + 1
        plt.imshow(y, vmin = norm_c_min, vmax = norm_c_max, cmap='viridis', interpolation='nearest')
        plt.colorbar(fraction=0.04, pad=0.04).set_label(label = 'Flux deviation from pixel time-median', size = fs)
    elif log_tf:
        plt.imshow(y, vmin = 0.1, vmax = c_max, cmap='viridis', interpolation='nearest', norm=colors.LogNorm())
        plt.colorbar(fraction=0.04, pad=0.04).set_label(label = 'Flux', size = fs)
    else:
        plt.imshow(y, vmin = c_min, vmax = c_max, cmap='viridis', interpolation='nearest') 
        plt.colorbar(fraction=0.04, pad=0.04).set_label(label = 'Flux', size = fs)
    ax = fig.axes[0]
    #mask = np.ma.masked_where(tpf.pipeline_mask, np.ones(np.shape(tpf[0].hdu[1].data['FLUX'][0,:,:])))
    #plt.imshow(mask, cmap=plt.cm.gray, interpolation='nearest', alpha = 1.0, label = 'Aperature mask') # this workd, but can only mask as a solid black box
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if not tpf.pipeline_mask[i, j]:
                if i == 0 and j == 0:
                    rect = patches.Rectangle(
                        #xy=(j + self.column - 0.5, i + self.row - 0.5),
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        color= 'k',
                        fill=False,
                        hatch="//",
                        label = 'Aperature mask')
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle(
                        #xy=(j + self.column - 0.5, i + self.row - 0.5),
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        color= 'k',
                        fill=False,
                        hatch="//",)
                    ax.add_patch(rect)
    plt.xlabel('Pixel column index', fontsize = fs); plt.ylabel('Pixel row index', fontsize = fs)
    plt.title(planet_name + ' transit ' + str(transit) + ', t = ' + str(np.round(t,3)) + ' days', fontsize = fs)
    plt.legend(loc = 'lower left', framealpha = 1.0)
    #print(ax.get_figure())
    return fig

for i in range(len(names[start:stop])):

    planet_name = names[i]
    star = planet_name.split(' ')[0]
    transit = transits[i]
    
    print(planet_name)
    print(star)
    
    # get window size
    query_name = planet_name
    if query_name == 'Kepler-1 b':  query_name = 'TrES-2 b'
    if query_name == 'Kepler-458 b': query_name = 'KIC 9663113 b'
    if query_name == 'Kepler-324 d': query_name = 'KOI-1831 d'    
    if query_name == 'Kepler-1703 b':  query_name = 'KOI-3503 b'
    if query_name == 'Kepler-968 d':  query_name = 'KOI-1833 d'
    if query_name == 'Kepler-460 b':  query_name = 'KIC 5437945 b'
    if query_name == 'Kepler-86 b':  query_name = 'PH2 b'
    if query_name == 'Kepler-1703 c':  query_name = 'KOI-3503 c'
    planet =  NASA_exoplanet_table[NASA_exoplanet_table['pl_name'] == query_name]
    if planet.empty:
        #print('tying alternate names.')
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
    nasa_duration = np.nanmedian(planet['pl_trandur']) / 24 # Planet transit duration [Days]

    results = pd.read_csv(results_dir + '/' + planet_name + '/Fit_data_' + planet_name + '.csv', skipfooter = 12, engine='python')
    fit_midpoint = results.iloc[transit,:]['Fit_transit_times']
    
    # get this from NASA reported duration and fit midpoint
    tmin = fit_midpoint - 5 * nasa_duration
    tmax = fit_midpoint + 5 * nasa_duration

    # you can't stitch target pixel files! So, use this to find which quarter to download
    tf = False
    #i = 0
    for quarter in lk.search_targetpixelfile(star, author="Kepler", cadence="long").download_all():
        #print(i); i += 1
        time = quarter.time.value
        if (tmin > time[0]) and (tmax < time[-1]):
            # region around transit we are interested in 
            tpf = quarter
            interval = (time >= tmin) * (time <= tmax)
            indxs = np.where(interval)[0] 
            tf = True
            break

    if tf == False: print('Could not find time interval, perhaps because spans two quarters!')

    #lc = lk.search_lightcurve(star, author="Kepler", cadence="long").download_all().stitch()
    time = tpf.time.value
    
    # for testing
    print('t_start:' + str(tmin))
    print('t_start:' + str(tmax)) 

    # define values for colorbar scale
    c_max = 0
    c_min = np.inf
    pixel_sum = np.zeros(np.shape(tpf[0].hdu[1].data['FLUX'][0,:,:]))
    for i in range(len(time[interval])):
        t = time[interval][i]
        this_max = np.nanmax(tpf[int(indxs[i])].hdu[1].data['FLUX'][0,:,:])
        this_min = np.nanmin(tpf[int(indxs[i])].hdu[1].data['FLUX'][0,:,:])
        pixel_sum += tpf[int(indxs[i])].hdu[1].data['FLUX'][0,:,:]
        if this_max > c_max: 
            c_max = this_max 
        if this_min < c_min: 
            c_min = this_min

    pixel_median = pixel_sum / len(time[interval])
    #plt.imshow(pixel_median); plt.colorbar()

    for i in range(len(time[interval])):
        #if i % 25 == 0: print(i)
        t = time[interval][i]
        fig = plot_tpf(tpf, i, c_min, c_max, planet, transit, t, norm_tf = norm_tf, log_tf = log_tf) # use my plotting

        # or try by saving each fig. Annoying but maybe the easiest bet.
        if norm_tf: outdir = star + '_TPFs_mednorm'
        elif log_tf: outdir = star + '_TPFs_logscale'   
        else: outdir = star + '_TPFs'
        if save:
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            fig.savefig(outdir + '/TPF' + str(i) + '.png')

    if save:
        if norm_tf: outfile = gif_dir + '/' + planet_name + '_t' + str(transit) + '_mednorm.gif'
        elif log_tf: outfile = gif_dir + '/' + planet_name + '_t' + str(transit) + '_logscale.gif' 
        else: outfile = gif_dir + '/' + planet_name + '_t' + str(transit) + '.gif'
        with imageio.get_writer(outfile , mode='I') as writer:
            for filename in os.listdir(outdir):
                if not filename.endswith('.png'): continue
                #print(filename)
                image = imageio.imread(outdir + '/' + filename)
                writer.append_data(image)