import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import glob
from branca.colormap import linear, LinearColormap
from backend import add_lapse_rate#, find_min_max
from shapely.geometry import Point, Polygon
import geopandas as gpd

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize':'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize': 'xx-large',
         'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

# Compability between pandas versions and mpl
pd.plotting.register_matplotlib_converters()

year = 2014
min_or_max = 'max'
version = '02P1'

for month in np.arange(1,13,1):
    start_time = datetime(year,month,1)
    end_time   = datetime(year,month+1,1)
    
    # Find all files composing the date range
    month_start = start_time.month
    month_end   = end_time.month
    month_range = np.arange(month_start, month_end+1, 1)
        
    if month_end < 10:
        month_range = [ '0'+str(m) for m in month_range ]
    else:
        month_temp = []
        for month in month_range:
            if month < 10:
                month_temp.append('0'+str(month))
            else:
                month_temp.append(str(month))
        month_range = month_temp
    
    filename_map_list = [ 'data/map-'+version+'-'+str(year)+m+'.nc' for m in month_range ]
    
    # Read data
    for filename_map in filename_map_list:
        ds_temp = xr.open_dataset(filename_map)
    
        if filename_map == filename_map_list[0]:
            ds = ds_temp
        else:
            ds = xr.concat([ds, ds_temp], dim="time")
    
    # Choose correct variable and date range
    diff = ds['t'+min_or_max+'_diff']
    diff = diff.sel(time=slice(start_time, end_time )) # Chose all hours in current date
    
    diff_average = diff.mean(dim="time").to_numpy()
    
    lon = ds['lon']
    lat = ds['lat']
    
    # Plot in map
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
    
    # NA
    lat_min = 20
    lat_max = 72
    lon_min = 360-140
    lon_max = 360-50
    
    # BC
    #lat_min = 46
    #lat_max = 58
    #lon_min = 360-135
    #lon_max = 360-112

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs.PlateCarree())
    
    resol = '10m'  # use data at this scale
    states = NaturalEarthFeature(category="cultural", scale=resol, facecolor="none", name="admin_1_states_provinces_shp")
    bodr = NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    land = NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor="none")
    ocean = NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor="none")
    lakes = NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='k', facecolor="none")
    rivers = NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='k', facecolor='none')
    
    ax.add_feature(land)
    ax.add_feature(ocean, linewidth=0.2 )
    ax.add_feature(lakes)
    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
    
    levels = [-10, -8, -6, -4, -2, 2, 4, 6, 8, 10]
    colormap = plt.get_cmap('bwr')
    norm = mcolors.BoundaryNorm(levels, ncolors=colormap.N, clip=True)
    
    cf = ax.pcolormesh(lon, lat, diff_average, transform=crs.PlateCarree(), cmap=colormap, norm=norm)
    
    cb = plt.colorbar(cf, orientation='horizontal', pad=0, aspect=50, extendrect=True)
    cb.set_label('diff T'+min_or_max+' (C)')
    
    if month == 1:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for January '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'01_NA.png')
    elif month == 2:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for Febuary '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'02_NA.png')
    elif month == 3:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for March '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'03_NA.png')
    elif month == 4:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for April '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'04_NA.png')
    elif month == 5:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for May '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'05_NA.png')
    elif month == 6:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for June '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'06_NA.png')
    elif month == 7:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for July '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'07_NA.png')
    elif month == 8:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for August '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'08_NA.png')
    elif month == 9:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for September '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'09_NA.png')
    elif month == 10:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for October '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'10_NA.png')
    elif month == 11:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for November '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'11_NA.png')
    elif month == 12:
        plt.title('Differences between RDRS v2.1 and ERA5_land \n Daily T'+min_or_max+' - Average for December '+str(year))
        plt.savefig('RDRS'+version+'-ERA5_T'+min_or_max+'_'+str(year)+'12_NA.png')

#plt.show()

