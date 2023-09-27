import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from backend import add_lapse_rate#, find_min_max

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize':'large',
         #'axes.titlesize':'large',
         'xtick.labelsize': 'large',
         'ytick.labelsize': 'large'}
pylab.rcParams.update(params)

# Compability between pandas versions and mpl
pd.plotting.register_matplotlib_converters()

def find_min_max(df, date_list):

    def func(val):
        minimum_val = df_copy[val['date_from'] : val['date_to']]['TT'].min()
        maximum_val = df_copy[val['date_from'] : val['date_to']]['TT'].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

    #if 'TT' in df.columns:
    try:
        df_temp = pd.DataFrame()
        df_temp['date_from'] = date_list
        df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

        df_copy = df.copy()
        df_copy.set_index('date', inplace=True)
        df_copy = pd.concat(list(df_temp.apply(func, axis=1)))

    #else:
    except:
        df_copy = df.copy()
        mask = (df_copy['date'] >= date_list[0]) & (df_copy['date'] <= date_list[-1])
        df_copy = df_copy.loc[mask]
        df_copy['date_from'] = df_copy['date']

        df_copy.set_index('date', inplace=True)

    return df_copy

def find_month(month):
    if month == 'January':     month_number = 1
    elif month == 'Febuary':   month_number = 2
    elif month == 'March':     month_number = 3
    elif month == 'April':     month_number = 4
    elif month == 'May':       month_number = 5
    elif month == 'June':      month_number = 6
    elif month == 'July':      month_number = 7
    elif month == 'August':    month_number = 8
    elif month == 'September': month_number = 9
    elif month == 'October':   month_number = 10
    elif month == 'November':  month_number = 11
    elif month == 'December':  month_number = 12

    return month_number

def make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, hour_range):

    sd_or_gradTT = 'SD'

    # Dates
    date_debut = year+'-01-02'
    date_fin   = year+'-12-14'
    date_list = pd.date_range(start=date_debut, end=date_fin)
 
    # Observations
    df_station_og = pd.read_pickle("data/"+clicked_id+"-station.pkl")

    df_station_sd = pd.DataFrame()
    if sd_or_gradTT in df_station_og.columns:
        df_station_sd['date'] = df_station_og['date']
        df_station_sd[sd_or_gradTT]   = df_station_og[sd_or_gradTT]
        mask = (df_station_sd['date'] > date_debut) & (df_station_sd['date'] <= date_fin)
        df_station_sd = df_station_sd.loc[mask]
        df_station_sd = df_station_sd[df_station_sd[sd_or_gradTT].notna()]

    df_station = find_min_max(df_station_og, date_list)

    if df_station.empty:
        station = False
    else:
        station = True

    # RDRS v2.1
    rdrs_02p1 = False
    df_rdrs_02p1_sd = pd.DataFrame()
    try:
        df_rdrs_02p1 = pd.read_pickle("data/"+hour_range+"/"+clicked_id+"-RDRSv02P1.pkl")
    
        df_rdrs_0p21 = df_rdrs_02p1.drop_duplicates(subset='date')
        elevation_rdrs = df_rdrs_02p1['elev'].loc[0]
    
        df_rdrs_02p1_sd = pd.DataFrame()
        if sd_or_gradTT in df_rdrs_02p1.columns:
            df_rdrs_02p1_sd['date'] = df_rdrs_02p1['date']
            df_rdrs_02p1_sd[sd_or_gradTT]   = df_rdrs_02p1[sd_or_gradTT]
            mask = (df_rdrs_02p1_sd['date'] > date_debut) & (df_rdrs_02p1_sd['date'] <= date_fin)
            df_rdrs_02p1_sd = df_rdrs_02p1_sd.loc[mask]
    
        df_rdrs_02p1 = find_min_max(df_rdrs_02p1, date_list)
    
        rdrs_02p1 = True

    except:
        rdrs_02p1 = False

    # RDRS v3
    rdrs_03test = False
    df_rdrs_03test_sd = pd.DataFrame()
    try:
        df_rdrs_03test = pd.read_pickle("data/"+hour_range+"/"+clicked_id+"-RDRSv3TEST.pkl")

        df_rdrs_03test = df_rdrs_03test.drop_duplicates(subset='date')
        elevation_rdrs = df_rdrs_03test['elev'].loc[0]

        df_rdrs_03test_sd = pd.DataFrame()
        if sd_or_gradTT in df_rdrs_03test.columns:
            df_rdrs_03test_sd['date'] = df_rdrs_03test['date']
            df_rdrs_03test_sd[sd_or_gradTT]   = df_rdrs_03test[sd_or_gradTT]
            mask = (df_rdrs_03test_sd['date'] > date_debut) & (df_rdrs_03test_sd['date'] <= date_fin)
            df_rdrs_03test_sd = df_rdrs_03test_sd.loc[mask]

        df_rdrs_03test = find_min_max(df_rdrs_03test, date_list)

        rdrs_03test = True

    except:
        rdrs_03test = False

    # RDRS v3IC405
    rdrs_ic405 = False
    df_rdrs_ic405_sd = pd.DataFrame()
    try:
        df_rdrs_ic405 = pd.read_pickle("data/"+hour_range+"/"+clicked_id+"-RDRSvic405.pkl")

        df_rdrs_ic405 = df_rdrs_ic405.drop_duplicates(subset='date')
        elevation_rdrs = df_rdrs_ic405['elev'].loc[0]

        df_rdrs_ic405_sd = pd.DataFrame()
        if sd_or_gradTT in df_rdrs_ic405.columns:
            df_rdrs_ic405_sd['date']       = df_rdrs_ic405['date']
            df_rdrs_ic405_sd[sd_or_gradTT] = df_rdrs_ic405[sd_or_gradTT]
            mask = (df_rdrs_ic405_sd['date'] > date_debut) & (df_rdrs_ic405_sd['date'] <= date_fin)
            df_rdrs_ic405_sd = df_rdrs_ic405_sd.loc[mask]

        df_rdrs_ic405 = find_min_max(df_rdrs_ic405, date_list)
        print(df_rdrs_ic405)
        print(df_rdrs_ic405_sd)

        rdrs_ic405 = True

    except:
        rdrs_ic405 = False

    # Lapse rate
    lapse_rate_rdrs = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_rdrs)
    lapse_rate_rdrs = np.array(lapse_rate_rdrs)

    # ERA5
    try:
        df_era5 = pd.read_pickle("data/"+clicked_id+"-ERA5.pkl")
        elevation_era5 = df_era5['elev'].iloc[0]

        df_era5_sd = pd.DataFrame()
        if sd_or_gradTT in df_era5.columns:
            df_era5_sd['date'] = df_era5['date']
            df_era5_sd[sd_or_gradTT]   = df_era5[sd_or_gradTT]
            mask = (df_era5_sd['date'] > date_debut) & (df_era5_sd['date'] <= date_fin)
            df_era5_sd = df_era5_sd.loc[mask]
            df_era5_sd = df_era5_sd[df_era5_sd[sd_or_gradTT].notna()]

        df_era5 = find_min_max(df_era5, date_list)

        # Lapse rate
        lapse_rate_era5 = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_era5)
        lapse_rate_era5 = np.array(lapse_rate_era5)

        era5 = True

        if df_era5.empty:
            elevation_era5 = 0.
            era5 = False

    except:
        elevation_era5 = 0.
        era5 = False

    # Plot
    temp_station = np.array(df_station[min_or_max].to_list()) 
 
    if rdrs_02p1:
        date = df_rdrs_02p1['date_from'].to_list()
        temp_rdrs_02p1 = np.array(df_rdrs_02p1[min_or_max].to_list())

    if rdrs_03test:
        date = df_rdrs_03test['date_from'].to_list()
        temp_rdrs_03test = np.array(df_rdrs_03test[min_or_max].to_list())

    if rdrs_ic405:
        date = df_rdrs_ic405['date_from'].to_list()
        temp_rdrs_ic405 = np.array(df_rdrs_ic405[min_or_max].to_list())

    if era5:
        temp_era5 = np.array(df_era5[min_or_max].to_list())

    biais = 0.

    fig, ax1 = plt.subplots(figsize=(10,5))

    lns = []
    if station: 
        tmax_obs  = ax1.plot(date, temp_station, 'k', label=min_or_max+' obs')
        lns = tmax_obs

    if rdrs_02p1:
        tmax_rdrs_02p1 = ax1.plot(date, (temp_rdrs_02p1 + lapse_rate_rdrs), 'b', label=min_or_max+' RDRS v2.1')
        lns = lns + tmax_rdrs_02p1
        print(lns)

    if rdrs_03test:
        tmax_rdrs_03test = ax1.plot(date, (temp_rdrs_03test + lapse_rate_rdrs), 'r', label=min_or_max+' RDRS v3')
        lns = lns + tmax_rdrs_03test

    if rdrs_ic405:
        tmax_rdrs_ic405 = ax1.plot(date, (temp_rdrs_ic405 + lapse_rate_rdrs), 'darkviolet', label=min_or_max+' RDRS IC405')
        lns = lns + tmax_rdrs_ic405

    if era5: 
        tmax_era5 = ax1.plot(date, (temp_era5 + lapse_rate_era5), 'g', label=min_or_max+' ERA5')
        lns = lns + tmax_era5

    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])

    if not df_station_sd.empty or not df_rdrs_02p1_sd.empty or not df_rdrs_03test_sd.empty or not df_era5_sd.empty or not df_rdrs_ic405_sd.empty:
        ax2 = ax1.twinx()
        sd = ax2.plot([], [], '--', color='gray', label="SD")
        lns = lns + sd

        if sd_or_gradTT == 'SD':
            ax2.set_ylabel('Snow depth [cm]')
            ax2.set_ylim([-5,500])
        elif sd_or_gradTT == 'gradTT':
            ax2.set_ylabel('grad TT [K/m]')
            ax2.set_ylim([-1.5,5.5])
        if not df_station_sd.empty:
            sd_obs  = ax2.plot(df_station_sd['date'], df_station_sd[sd_or_gradTT], '--k', label=sd_or_gradTT+' obs')
            #lns = lns + sd_obs
 
        if not df_rdrs_02p1_sd.empty:
            sd_rdrs = ax2.plot(df_rdrs_02p1_sd['date'],    df_rdrs_02p1_sd[sd_or_gradTT], '--b', label=sd_or_gradTT+' RDRS')
            #lns = lns + sd_rdrs
    
        if not df_rdrs_03test_sd.empty:
            sd_rdrs = ax2.plot(df_rdrs_03test_sd['date'],    df_rdrs_03test_sd[sd_or_gradTT], '--r', label=sd_or_gradTT+' RDRS')
            #lns = lns + sd_rdrs
    
        if not df_rdrs_ic405_sd.empty:
            sd_rdrs = ax2.plot(df_rdrs_ic405_sd['date'],    df_rdrs_ic405_sd[sd_or_gradTT], '--m', label=sd_or_gradTT+' RDRS')

        if era5 and not df_era5_sd.empty:
            sd_era5 = ax2.plot(df_era5_sd['date'],    df_era5_sd[sd_or_gradTT], '--g', label=sd_or_gradTT+' ERA5')
            #lns = lns + sd_era5
    
    ax1.grid(True)

    # added these three lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.02,1), borderaxespad=0, loc='upper left')

    plt.title(min_or_max+' at '+clicked_name)

    return fig, elevation_rdrs, elevation_era5, biais

########
# Page #
########

year = "2014"

timeserie_or_diurnal = 'timeserie'
hour_range = '06-17'

lapse_type = 'none'
min_or_max = 'Tmin'

version = ''

df_station_info = pd.read_csv('data/station-biais-eccc.obs', delim_whitespace=True, skiprows=2)

#"BELLA COOLA A" 381 1060841 52.39 -126.6 35.7
clicked_id   = '1060841'
clicked_name = 'BELLA COOLA A'
clicked_elev = 35.7
    
fig, elevation_rdrs, elevation_era5, biais = make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, hour_range)

plt.show()
     
