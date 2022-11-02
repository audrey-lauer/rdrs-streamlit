import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
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

matplotlib.use("agg")
_lock = RendererAgg.lock

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

def make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, sd_or_gradTT):
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
    if '02P1' in version:
        try:
            df_rdrs_02p1 = pd.read_pickle("data/"+clicked_id+"-RDRSv02P1.pkl")
    
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
    if '3TEST' in version:
        try:
            df_rdrs_03test = pd.read_pickle("data/"+clicked_id+"-RDRSv3TEST.pkl")

            df_rdrs_0p21 = df_rdrs_03test.drop_duplicates(subset='date')
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

    # GDRS
    try:
        df_gdrs = pd.read_pickle("data/"+clicked_id+"-GDRSv"+version+".pkl")
        df_gdrs = df_gdrs.drop_duplicates(subset='date')

        df_gdrs_sd = pd.DataFrame()
        if sd_or_gradTT in df_gdrs.columns:
            df_gdrs_sd['date'] = df_gdrs['date']
            df_gdrs_sd[sd_or_gradTT]   = df_gdrs[sd_or_gradTT]
            mask = (df_gdrs_sd['date'] > date_debut) & (df_gdrs_sd['date'] <= date_fin)
            df_gdrs_sd = df_gdrs_sd.loc[mask]
            df_gdrs_sd = df_gdrs_sd[df_gdrs_sd[sd_or_gradTT].notna()]

        df_gdrs = find_min_max(df_gdrs, date_list)

        # Lapse rate
        #lapse_rate_gdrs = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_gdrs)
        #lapse_rate_gdrs = np.array(lapse_rate_gdrs)

        gdrs = True

    except:
        gdrs = False

    # Plot
    temp_station = np.array(df_station[min_or_max].to_list()) 
 
    if rdrs_02p1:
        date = df_rdrs_02p1['date_from'].to_list()
        temp_rdrs_02p1 = np.array(df_rdrs_02p1[min_or_max].to_list())

    if rdrs_03test:
        date = df_rdrs_03test['date_from'].to_list()
        temp_rdrs_03test = np.array(df_rdrs_03test[min_or_max].to_list())

    if era5:
        temp_era5 = np.array(df_era5[min_or_max].to_list())

    if gdrs:
        temp_gdrs = np.array(df_gdrs[min_or_max].to_list())

    #biais = (temp_rdrs_max + lapse_rate_rdrs) - temp_station_max
    biais = 0.

    fig, ax1 = plt.subplots(figsize=(10,5))

    if station: 
        tmax_obs  = ax1.plot(date, temp_station, 'k', label=min_or_max+' obs')
        lns = tmax_obs

    if rdrs_02p1:
        tmax_rdrs_02p1 = ax1.plot(date, (temp_rdrs_02p1 + lapse_rate_rdrs), 'b', label=min_or_max+' RDRS v2.1')
        lns = lns + tmax_rdrs_02p1

    if rdrs_03test:
        tmax_rdrs_03test = ax1.plot(date, (temp_rdrs_03test + lapse_rate_rdrs), 'r', label=min_or_max+' RDRS v3')
        lns = lns + tmax_rdrs_03test

    if era5: 
        tmax_era5 = ax1.plot(date, (temp_era5 + lapse_rate_era5), 'g', label=min_or_max+' ERA5')
        lns = lns + tmax_era5

    #if gdrs: 
    #    tmax_gdrs = ax1.plot(date, (temp_gdrs), 'm', label=min_or_max+' GDRS')
    #    lns = lns + tmax_gdrs

    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])

    #if firstlevel:
    #    tmax_rdrs_1stlevel = ax1.plot(date, np.array(df_rdrs_1stlevel[min_or_max].to_list()), 'c', label='1st level RDRS')
    #    lns = lns + tmax_rdrs_1stlevel

    if not df_station_sd.empty or not df_rdrs_02p1_sd.empty or not df_rdrs_03test_sd.empty or not df_era5_sd.empty or not df_gdrs_sd.empty:
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
    
        if era5 and not df_era5_sd.empty:
            sd_era5 = ax2.plot(df_era5_sd['date'],    df_era5_sd[sd_or_gradTT], '--g', label=sd_or_gradTT+' ERA5')
            #lns = lns + sd_era5
    
        if gdrs and not df_gdrs_sd.empty:
            sd_gdrs = ax2.plot(df_gdrs_sd['date'],    df_gdrs_sd[sd_or_gradTT], '--m', label=sd_or_gradTT+' GDRS')
            #lns = lns + sd_gdrs

    ax1.grid(True)

    # added these three lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.02,1), borderaxespad=0, loc='upper left')

    plt.title(min_or_max+' at '+clicked_name)

    return fig, elevation_rdrs, elevation_era5, biais

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
        version_02p1   = st.checkbox('RDRS v2.1')
        version_03test = st.checkbox('RDRS v3')

        version = []
        if version_02p1: version.append('02P1')
        if version_03test: version.append('3TEST')

        if dataset == 'ECCC network':
            if '02P1' in version:
                year = st.slider('Pick the year', 1990, 2018)
                year = str(year)
            else:
                year = st.slider('Pick the year', 2014, 2015)
                year = str(year)
        elif dataset == 'BC archive':
            year = st.radio('Pick the year',['2017','2018'])
        elif dataset == 'Wood':
            year = st.radio('Pick the year',['2005','2006', '2007','2008','2009','2010'])
        else:
            year = st.radio('Pick the year',['1990','1996', '2017','2018'])

        lapse_type = st.radio('Lapse rate type',['none','fixed','Stahl'])
        min_or_max = st.radio('Tmin or Tmax?',['Tmin','Tmax'])
        sd_or_gradTT = st.radio('SD or gradient?',['SD','gradTT'])

    if dataset == 'ECCC network':
        df_station_info = pd.read_csv('data/station-biais-eccc.obs', delim_whitespace=True, skiprows=2)
    elif dataset == 'BC archive':
        df_station_info = pd.read_csv('data/station-biais-canswe.obs', delim_whitespace=True, skiprows=2)
    elif dataset == 'Wood':
        df_station_info = pd.read_csv('data/station-biais-wood.obs', delim_whitespace=True, skiprows=2)

    main_map = make_map(df_station_info, 'DATA.BIAIS_'+year+'_v'+version[0])

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
    
            fig, elevation_rdrs, elevation_era5, biais = make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, sd_or_gradTT)
     
            df_elev = pd.DataFrame(index=['Station','RDRS','ERA5-land'])
            df_elev['Elevation (m)'] = [clicked_elev, elevation_rdrs, elevation_era5]
            st.dataframe(df_elev)
    
            st.write(fig)

