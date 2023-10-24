import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from streamlit_folium import folium_static, st_folium
import folium
from branca.colormap import linear, LinearColormap
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

# Wide streamlit page
st.set_page_config(layout="wide")

@st.cache(hash_funcs={folium.folium.Map: lambda _: None}, allow_output_mutation=True)
def make_map(df_station_info, field_to_color_by):
    main_map = folium.Map(location=(52, -121), zoom_start=5)
    colormap = linear.RdYlBu_11.scale(-5,5)
    colormap.caption = 'Yearly bias'
    colormap.add_to(main_map)

    for i in df_station_info.index:
        lat  = df_station_info['LAT'].loc[i]
        lon  = df_station_info['LON'].loc[i]
        elev = df_station_info['ELEV'].loc[i]
        name = df_station_info['ID'].loc[i]

        if np.isnan( df_station_info[field_to_color_by].loc[i] ):
            continue
        elif df_station_info[field_to_color_by].loc[i] < -900:
            continue
        else:
            icon_color = colormap(df_station_info[field_to_color_by].loc[i])

            folium.CircleMarker(location=[lat, lon],
                        fill=True,
                        fill_color=icon_color,
                        color='black',
                        fill_opacity=0.8,
                        radius=5,
                        popup=name,
                        ).add_to(main_map)

    return main_map

def find_min_max(df, date_list, variable):

    def func(val):
        minimum_val = df_copy[val['date_from'] : val['date_to']][variable].min()
        maximum_val = df_copy[val['date_from'] : val['date_to']][variable].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

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

    df_station = find_min_max(df_station_og, date_list,'TT')

    if df_station.empty:
        station = False
    else:
        station = True

    df_rdrs    = dict.fromkeys(version)
    df_rdrs_sd = dict.fromkeys(version)
    df_rdrs_tt = dict.fromkeys(version)
    df_rdrs_t2 = dict.fromkeys(version)
    rdrs_sd    = dict.fromkeys(version, False)
    rdrs_tt    = dict.fromkeys(version, False)
    rdrs_t2    = dict.fromkeys(version, False)
    elev_rdrs  = dict.fromkeys(version, 0)

    for v in version:
        #try:
        df_rdrs[v]    = pd.DataFrame()
        df_rdrs_sd[v] = pd.DataFrame()
        df_rdrs_tt[v] = pd.DataFrame()
        df_rdrs_t2[v] = pd.DataFrame()

        df_rdrs[v]   = pd.read_pickle("data/"+hour_range+"/"+clicked_id+"-RDRSv"+v+".pkl")
        df_rdrs[v]   = df_rdrs[v].drop_duplicates(subset='date')
        #df_rdrs[v]   = df_rdrs[v].reset_index()
        elev_rdrs[v] = df_rdrs[v]['elev'].loc[0]

        print(df_rdrs[v])

        df_rdrs_tt[v] = find_min_max(df_rdrs[v], date_list, 'TT')
        print(v)
        print(df_rdrs_tt[v])

        # SD
        if 'SD' in df_rdrs[v].columns:
            df_rdrs_sd[v]['date'] = df_rdrs[v]['date']
            df_rdrs_sd[v]['SD']   = df_rdrs[v]['SD']
            mask = ( df_rdrs_sd[v]['date'] > date_debut ) & ( df_rdrs_sd[v]['date'] <= date_fin )
            df_rdrs_sd[v] = df_rdrs_sd[v].loc[mask]

            rdrs_sd[v] = True

        try:
            df_rdrs_t2[v] = find_min_max(df_rdrs[v], date_list, 'T2')
            rdrs_t2[v] = True
        except:
            print('No T2 in experience '+v)
            continue

        rdrs_tt[v] = True

        #except:
        #    print('Bug in experience '+v)
        #    continue

    # Lapse rate
    lapse_rate_rdrs = dict.fromkeys(version)
    for v in version:
        lapse_rate_rdrs[v] = add_lapse_rate(lapse_type, date_list, clicked_elev, elev_rdrs[v])
        lapse_rate_rdrs[v] = np.array(lapse_rate_rdrs[v])

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

        df_era5 = find_min_max(df_era5, date_list, 'TT')

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
    date_station = df_station['date_from'].to_list()
    temp_station = np.array(df_station[min_or_max].to_list()) 

    date_rdrs = dict.fromkeys(version)
    tt_rdrs   = dict.fromkeys(version)
    t2_rdrs   = dict.fromkeys(version)
    for v in version:
        date_rdrs[v] = df_rdrs_tt[v]['date_from'].to_list()
        tt_rdrs[v]   = np.array(df_rdrs_tt[v][min_or_max].to_list())

        #if rdrs_t2[v]:
        try:
            t2_rdrs[v] = np.array(df_rdrs_t2[v][min_or_max].to_list())
        except:
            continue
    
        date = date_rdrs[v]
 
    if era5:
        date_era5 = df_era5['date_from'].to_list()
        temp_era5 = np.array(df_era5[min_or_max].to_list())

    biais = 0.

    color = {
        '02P1' : 'b',
        '3TEST': 'r',
        'ic401'     : 'magenta',
        'ic401wCWA' : 'coral',
        'ic404'     : 'slateblue',
        'ic405'     : 'darkviolet',
        'ic406'     : 'darkmagenta',
        'ic406w8'   : 'deeppink',
        'ic406w9'   : 'palevioletred'
    }

    fig, ax1 = plt.subplots(figsize=(10,5))

    lns = []
    # TT
    if station: 
        tmax_obs  = ax1.plot(date_station, temp_station, 'k', label=min_or_max+' obs')
        lns = tmax_obs

    for v in version:
        tmax_rdrs = ax1.plot(date_rdrs[v], (tt_rdrs[v] + lapse_rate_rdrs[v]), color[v], label=min_or_max+' RDRS '+v)
        lns = lns + tmax_rdrs

    if era5: 
        tmax_era5 = ax1.plot(date_era5, (temp_era5 + lapse_rate_era5), 'g', label=min_or_max+' ERA5')
        lns = lns + tmax_era5

    # T2
    t2 = ax1.plot([], [], ':', color='gray', label="T2")
    lns = lns + t2 

    for v in version:
        #if rdrs_t2[v]:
        try:
            t2_rdrs = ax1.plot(date_rdrs[v], df_rdrs_t2[v], ':', color=color[v])
        except:
            continue

    # SD
    ax2 = ax1.twinx()
    sd = ax2.plot([], [], '--', color='gray', label="SD")
    lns = lns + sd

    ax2.set_ylabel('Snow depth [cm]')
    ax2.set_ylim([-5,500])

    if not df_station_sd.empty:
        sd_obs  = ax2.plot(df_station_sd['date'], df_station_sd[sd_or_gradTT], '--k')

    for v in version:
        if rdrs_sd[v]:
            sd_rdrs = ax2.plot(df_rdrs_sd[v]['date'], df_rdrs_sd[v][sd_or_gradTT], '--', color=color[v])

    if era5 and not df_era5_sd.empty:
        sd_era5 = ax2.plot(df_era5_sd['date'],    df_era5_sd[sd_or_gradTT], '--g')
        
    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])
    ax1.grid(True)

    # added these three lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.02,1), borderaxespad=0, loc='upper left')

    plt.title(min_or_max+' at '+clicked_name)

    return fig, elev_rdrs, elevation_era5, biais

########
# Page #
########
st.write('Investigation of temperature bias in BC')

dataset = st.selectbox('Observation dataset',['ECCC network'])
#dataset = st.selectbox('Observation dataset',['ECCC network','BC archive','Wood','RDRS - ERA5_land'])

if dataset == 'ECCC network' or dataset == 'BC archive' or dataset == 'Wood':

    # 3 columns
    # 1st column: map
    # 2nd column: parameters
    # 3rd column: timeserie
    col1, col2, col3 = st.columns([0.3,0.7,1])

    with col1:
        st.header("Parameters")
        st.write("Choose the parameters for timeserie")

        #version = st.radio('Pick the RDRS version',['02P1','3TEST'])
        st.caption("Pick the RDRS version")
        version_01     = st.checkbox('RDRS v1')
        version_02p1   = st.checkbox('RDRS v2.1')
        version_03test = st.checkbox('RDRS v3')
        version_03Lmin = st.checkbox('RDRS v3 Lmin')
        version_03tdiag   = st.checkbox('RDRS v3 tdiaglim')
        version_ic401     = st.checkbox('RDRS vIC401')
        version_ic401wcwa = st.checkbox('RDRS vIC401wCWA')
        version_ic404     = st.checkbox('RDRS vIC404')
        version_ic405     = st.checkbox('RDRS vIC405')
        version_ic406     = st.checkbox('RDRS vIC406')
        version_ic406w8   = st.checkbox('RDRS vIC406w8')
        version_ic406w9   = st.checkbox('RDRS vIC406w9')
        version_rdps      = st.checkbox('RDPS')
        version_hrdps     = st.checkbox('HRDPS')

        version = []
        if version_02p1: version.append('02P1')
        if version_03test: version.append('3TEST')
        if version_03Lmin: version.append('3Lmin')
        if version_03tdiag: version.append('3tdiaglim')
        if version_ic401:     version.append('ic401')
        if version_ic401wcwa: version.append('ic401wCWA')
        if version_ic404:     version.append('ic404')
        if version_ic405:     version.append('ic405')
        if version_ic406:     version.append('ic406')
        if version_ic406w8:   version.append('ic406w8')
        if version_ic406w9:   version.append('ic406w9')
        if version_01: version.append('v1')
        if version_rdps: version.append('rdps')
        if version_hrdps: version.append('hrdps')

        if dataset == 'ECCC network':
            if '02P1' in version:
                year = st.slider('Pick the year', 1990, 2018)
                year = str(year)
            else:
                year = st.slider('Pick the year', 1992, 2014)
                year = str(year)
        elif dataset == 'BC archive':
            year = st.radio('Pick the year',['2017','2018'])
        elif dataset == 'Wood':
            year = st.radio('Pick the year',['2005','2006', '2007','2008','2009','2010'])
        else:
            year = st.radio('Pick the year',['1990','1996', '2017','2018'])

        timeserie_or_diurnal = st.radio('Timeserie or diurnal cycle?',['timeserie','diurnal cycle'])
        hour_range = st.radio('Hours?',['06-17','12-23'])

        if timeserie_or_diurnal == 'timeserie':
            lapse_type = st.radio('Lapse rate type',['none','fixed','Stahl'])
            min_or_max = st.radio('Tmin or Tmax?',['Tmin','Tmax'])
            #sd_or_gradTT = st.radio('SD or gradient?',['SD','gradTT'])

    if dataset == 'ECCC network':
        df_station_info = pd.read_csv('data/station-biais-eccc.obs')
    elif dataset == 'BC archive':
        df_station_info = pd.read_csv('data/station-biais-canswe.obs', delim_whitespace=True, skiprows=2)
    elif dataset == 'Wood':
        df_station_info = pd.read_csv('data/station-biais-wood.obs', delim_whitespace=True, skiprows=2)

    #main_map = make_map(df_station_info, 'DATA.BIAIS_'+year+'_v'+version[0])
    if year == '1992':
        main_map = make_map(df_station_info, 'DATA.BIAIS_1992'+'_v'+version[0])
    else:
        main_map = make_map(df_station_info, 'DATA.BIAIS_2014'+'_v'+version[0])

    with col2:
        st.header("Interactive map")
        st.write("Click on a station to generate timeserie")
        # Plot map and get data of last click/zoom/etc
        st_data = st_folium(main_map, width=500, height=500)
    
    if st_data['last_object_clicked'] is not None:
        clicked_lat = st_data['last_object_clicked']['lat']
        clicked_lon = st_data['last_object_clicked']['lng']
    
        clicked_info = df_station_info[(df_station_info['LAT'] == clicked_lat) & (df_station_info['LON'] == clicked_lon)]
        clicked_id   = clicked_info['NO'].to_list()[0]
        clicked_name = clicked_info['ID'].to_list()[0]
        clicked_elev   = clicked_info['ELEV'].to_list()[0]
    
        with col3:
            st.header("Timeserie")
    
            if timeserie_or_diurnal == 'timeserie':
                fig, elev_rdrs, elevation_era5, biais = make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, hour_range)
     
                df_elev = pd.DataFrame(index=['Station','RDRS','ERA5-land'])
                df_elev['Elevation (m)'] = [clicked_elev, elev_rdrs[version[0]], elevation_era5]
                st.dataframe(df_elev)

                st.write(fig)

            elif timeserie_or_diurnal == 'diurnal cycle':

                month = st.selectbox('Month',['January','Febuary','March','April','May','June','July','August','September','October','November','December','DJF','MAM','JJA','SON'])
                fig   = make_diurnal_cycle(year, clicked_id, clicked_name, version, month, hour_range)
           
                st.write(fig)

