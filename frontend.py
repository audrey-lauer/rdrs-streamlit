import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from streamlit_folium import folium_static, st_folium
import folium
from folium import plugins
from branca.colormap import linear, LinearColormap
from backend import add_lapse_rate#, find_min_max
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, Point

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
def make_map(df_station_info, field_to_color_by, draw_polygon):
    main_map = folium.Map(location=(52, -121), zoom_start=5)
    colormap = linear.RdYlBu_11.scale(-5,5)
    colormap.caption = 'Yearly bias'
    colormap.add_to(main_map)

    # Enable drawing polygons on the map
    if draw_polygon:
        draw_plugin = plugins.Draw(export=True)
        draw_plugin.add_to(main_map)

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

    df_temp = pd.DataFrame()
    df_temp['date_from'] = date_list
    df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

    df_copy = df.copy()
    df_copy.set_index('date', inplace=True)
    df_copy = pd.concat(list(df_temp.apply(func, axis=1)))

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
    if year == '1992':
        date_debut = '1992-01-02'
        date_fin   = '1992-12-14'
    else:
        date_debut = year+'-01-02'
        date_fin   = year+'-12-14'

    #date_fin   = year+'-05-14'
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

    df_station_copy = df_station_og.copy()
    mask = (df_station_copy['date'] >= date_list[0]) & (df_station_copy['date'] <= date_list[-1])
    df_station_copy = df_station_copy.loc[mask]
    df_station_copy['date_from'] = df_station_copy['date']
    df_station_copy.set_index('date', inplace=True)

    df_station = df_station_copy

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
        df_rdrs[v]    = pd.DataFrame()
        df_rdrs_sd[v] = pd.DataFrame()
        df_rdrs_tt[v] = pd.DataFrame()
        df_rdrs_t2[v] = pd.DataFrame()

        try:
            df_rdrs[v]   = pd.read_pickle("data/"+hour_range+"/"+clicked_id+"-RDRSv"+v+".pkl")
            df_rdrs[v]   = df_rdrs[v].drop_duplicates(subset='date')
            elev_rdrs[v] = df_rdrs[v]['elev'].loc[0]

            df_rdrs_tt[v] = find_min_max(df_rdrs[v], date_list, 'TT')
            rdrs_tt[v] = True
        except:
            continue

        # SD
        if 'SD' in df_rdrs[v].columns:
            df_rdrs_sd[v]['date'] = df_rdrs[v]['date']
            df_rdrs_sd[v]['SD']   = df_rdrs[v]['SD']
            mask = ( df_rdrs_sd[v]['date'] > date_debut ) & ( df_rdrs_sd[v]['date'] <= date_fin )
            df_rdrs_sd[v] = df_rdrs_sd[v].loc[mask]

            df_rdrs_sd[v].sort_values(by='date', inplace=True)

            rdrs_sd[v] = True

        try:
            df_rdrs_t2[v] = find_min_max(df_rdrs[v], date_list, 'T2')
            rdrs_t2[v] = True
        except:
            print('No T2 in experience '+v)
            continue

    # Lapse rate
    lapse_rate_rdrs = dict.fromkeys(version)
    for v in version:
        lapse_rate_rdrs[v] = add_lapse_rate(lapse_type, date_list, clicked_elev, elev_rdrs[v])
        lapse_rate_rdrs[v] = np.array(lapse_rate_rdrs[v])

    # ERA5
    era5 = False
    elevation_era5 = 0.
    df_era5    = pd.DataFrame()
    df_era5_sd = pd.DataFrame()
    print(version)
    if 'ERA5L' in version:
        try:
            df_era5 = pd.read_pickle("data/"+clicked_id+"-ERA5.pkl")
            elevation_era5 = df_era5['elev'].iloc[0]
    
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

    return elev_rdrs, elevation_era5, df_station, station, df_station_sd, df_rdrs_tt, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd

def make_figure(clicked_name, version, min_or_max, df_station, station, df_station_sd, df_rdrs_tt, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd):

    # Plot
    date_station = df_station['date_from'].to_list()
    temp_station = np.array(df_station[min_or_max].to_list()) 

    date_rdrs_tt = dict.fromkeys(version)
    date_rdrs_t2 = dict.fromkeys(version)
    tt_rdrs   = dict.fromkeys(version)
    t2_rdrs   = dict.fromkeys(version)
    for v in version:
        try:
            date_rdrs_tt[v] = df_rdrs_tt[v]['date_from'].to_list()
            tt_rdrs[v]   = np.array(df_rdrs_tt[v][min_or_max].to_list())
        except:
            rdrs_tt[v] = False
            continue

    if era5:
        date_era5 = df_era5['date_from'].to_list()
        temp_era5 = np.array(df_era5[min_or_max].to_list())

    biais = 0.

    color = {
        '02P1' : 'royalblue',
        '3TEST': 'r',
        'ic401'      : 'magenta',
        'ic401v3'    : 'purple',
        'ic402'      : 'lime',
        'ic401wCWA'  : 'mediumvioletred',
        'ic401wCHDSD': 'palevioletred',
        'ic404'      : 'slateblue',
        'ic405'      : 'darkviolet',
        'ic406'      : 'darkmagenta',
        'ic407'      : 'turquoise',
        'ic408'      : 'gold',
        'ic409'      : 'orange',
        'ic406w8'    : 'deeppink',
        'ic406w9'    : 'palevioletred',
        'ic411'      : 'darkblue',
        'ic414'      : 'indianred',
        'ic414H'     : 'firebrick',
        'ic415'      : 'teal',
        'ic416'      : 'orange',
        'ic417'      : 'tomato',
        'ic418'      : 'orange',
        'ic419'      : 'slateblue',
        'ic420'      : 'tomato',
        'ic421'      : 'orange',
        'ic422'      : 'red',
        'ic425'      : 'tomato'
    }

    fig, ax1 = plt.subplots(figsize=(10,5))

    lns = []
    # TT
    if station: 
        tmax_obs  = ax1.plot(date_station, temp_station, 'k', label=min_or_max+' obs')
        lns = tmax_obs

    for v in version:
        if rdrs_tt[v]:
            #tmax_rdrs = ax1.plot(date_rdrs_tt[v], (tt_rdrs[v] + lapse_rate_rdrs[v]), color[v], label=min_or_max+' RDRS '+v)
            tmax_rdrs = ax1.plot(date_rdrs_tt[v], tt_rdrs[v], color[v], label=min_or_max+' RDRS '+v)
            lns = lns + tmax_rdrs

    if era5: 
        tmax_era5 = ax1.plot(date_era5, temp_era5, 'g', label=min_or_max+' ERA5')
        lns = lns + tmax_era5

    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):d}'))
    ax1.grid(True)

    # SD
    ax2 = ax1.twinx()
    sd = ax2.plot([], [], '--', color='gray', label="SD")
    lns = lns + sd

    ax2.set_ylabel('Snow depth [cm]')
    ax2.set_ylim([-5,950])

    if not df_station_sd.empty:
        sd_obs  = ax2.plot(df_station_sd['date'], df_station_sd['SD'], '--k')

    for v in version:
        if rdrs_sd[v]:
            sd_rdrs = ax2.plot(df_rdrs_sd[v]['date'], df_rdrs_sd[v]['SD'], '--', color=color[v])

    if era5 and not df_era5_sd.empty:
        sd_era5 = ax2.plot(df_era5_sd['date'], df_era5_sd['SD'], '--g')
        
    # added these three lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.02,1), borderaxespad=0, loc='upper left')

    plt.title(min_or_max+' at '+clicked_name)
    
    return fig

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
    col1, col1_5, col2, col3 = st.columns([0.3,0.2,0.6,0.9])

    with col1:
        st.header("Experiences")

        #version = st.radio('Pick the RDRS version',['02P1','3TEST'])
        st.caption("Pick the RDRS version")
        version_era5   = st.checkbox('ERA5-land', True)
        version_01     = st.checkbox('RDRS v1')
        version_02p1   = st.checkbox('RDRS v2.1', True)
        version_03test = st.checkbox('RDRS v3')
        version_03Lmin = st.checkbox('RDRS v3 Lmin')
        version_03tdiag   = st.checkbox('RDRS v3 tdiaglim')
        version_ic401     = st.checkbox('RDRS vIC401')
        version_ic401v3   = st.checkbox('RDRS vIC401v3')
        version_ic401wcwa = st.checkbox('RDRS vIC401wCWA')
        version_ic401wchdsd = st.checkbox('RDRS vIC401wCHDSD')
        version_ic402     = st.checkbox('RDRS vIC402')
        version_ic404     = st.checkbox('RDRS vIC404')
        version_ic405     = st.checkbox('RDRS vIC405')
        version_ic406     = st.checkbox('RDRS vIC406')
        version_ic407     = st.checkbox('RDRS vIC407')
        version_ic408     = st.checkbox('RDRS vIC408')
        version_ic406w8   = st.checkbox('RDRS vIC406w8')
        version_ic409     = st.checkbox('RDRS vIC409')
        version_ic406w9   = st.checkbox('RDRS vIC406w9')
        version_ic411     = st.checkbox('RDRS vIC411')
        version_ic414     = st.checkbox('RDRS vIC414')
        version_ic414H    = st.checkbox('RDRS vIC414H')
        version_ic415     = st.checkbox('RDRS vIC415')
        version_ic416     = st.checkbox('RDRS vIC416')
        version_ic417     = st.checkbox('RDRS vIC417')
        version_ic418     = st.checkbox('RDRS vIC418')
        version_ic419     = st.checkbox('RDRS vIC419')
        version_ic420     = st.checkbox('RDRS vIC420')
        version_ic421     = st.checkbox('RDRS vIC421')
        version_ic422     = st.checkbox('RDRS vIC422')
        version_rdps      = st.checkbox('RDPS')
        version_hrdps     = st.checkbox('HRDPS')

        version = []
        if version_era5: version.append('ERA5L')
        if version_02p1: version.append('02P1')
        if version_03test: version.append('3TEST')
        if version_03Lmin: version.append('3Lmin')
        if version_03tdiag: version.append('3tdiaglim')
        if version_ic401:     version.append('ic401')
        if version_ic401v3:   version.append('ic401v3')
        if version_ic402:     version.append('ic402')
        if version_ic401wcwa: version.append('ic401wCWA')
        if version_ic401wchdsd: version.append('ic401wCHDSD')
        if version_ic404:     version.append('ic404')
        if version_ic405:     version.append('ic405')
        if version_ic406:     version.append('ic406')
        if version_ic407:     version.append('ic407')
        if version_ic408:     version.append('ic408')
        if version_ic406w8:   version.append('ic406w8')
        if version_ic409:     version.append('ic409')
        if version_ic406w9:   version.append('ic406w9')
        if version_ic411:     version.append('ic411')
        if version_ic414:     version.append('ic414')
        if version_ic414H:    version.append('ic414H')
        if version_ic415:     version.append('ic415')
        if version_ic416:     version.append('ic416')
        if version_ic417:     version.append('ic417')
        if version_ic418:     version.append('ic418')
        if version_ic419:     version.append('ic419')
        if version_ic420:     version.append('ic420')
        if version_ic421:     version.append('ic421')
        if version_ic422:     version.append('ic422')
        if version_01: version.append('v1')
        if version_rdps: version.append('rdps')
        if version_hrdps: version.append('hrdps')

    with col1_5:
        st.header("")

        if dataset == 'ECCC network':
            year = st.selectbox('Pick the year', [1992,1993,2014])
            year = str(year)
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

        geometry = [Point(xy) for xy in zip(df_station_info['LON'], df_station_info['LAT'])]
        gdf_station_info = GeoDataFrame(df_station_info, geometry=geometry)

    elif dataset == 'BC archive':
        df_station_info = pd.read_csv('data/station-biais-canswe.obs', delim_whitespace=True, skiprows=2)
    elif dataset == 'Wood':
        df_station_info = pd.read_csv('data/station-biais-wood.obs', delim_whitespace=True, skiprows=2)

    main_map         = make_map(df_station_info, 'DATA.BIAIS_v02P1', True)
    #if year == '1992':
    #    main_map = make_map(df_station_info, 'DATA.BIAIS_1992'+'_v'+version[0])
    #else:
    #    main_map = make_map(df_station_info, 'DATA.BIAIS_2014'+'_v'+version[0])

    with col2:
        st.header("Interactive map")
        st.write("Click on a station to generate timeserie")
        # Plot map and get data of last click/zoom/etc
        st_data = st_folium(main_map, width=500, height=500)

    with col3:
        #tab1, tab2 = st.tabs(["station", "region"])
        st.header("Timeserie")
        average = st.radio('',['station','average'])

        #with tab1:
        if average == 'station':

            if st_data['last_object_clicked'] is not None:
                try:
                    clicked_lat = st_data['last_object_clicked']['lat']
                    clicked_lon = st_data['last_object_clicked']['lng']
            
                    clicked_info = df_station_info[(df_station_info['LAT'] == clicked_lat) & (df_station_info['LON'] == clicked_lon)]
                    clicked_id   = clicked_info['NO'].to_list()[0]
                    clicked_name = clicked_info['ID'].to_list()[0]
                    clicked_elev   = clicked_info['ELEV'].to_list()[0]
    
                    # Plot station figure
                    elev_rdrs, elevation_era5, df_station, station, df_station_sd, df_rdrs_tt, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd = make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, hour_range)
                    fig = make_figure(clicked_name, version, min_or_max, df_station, station, df_station_sd, df_rdrs_tt, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd)
     
                    df_elev = pd.DataFrame(index=['Station','RDRS','ERA5-land'])
                    df_elev['Elevation (m)'] = [clicked_elev, elev_rdrs[version[0]], elevation_era5]
                    st.dataframe(df_elev)

                    st.write(fig)

                except:
                    st.write('Click on a station')

        #with tab2:
        if average == 'average':

            if st_data['last_active_drawing'] is not None:
                try:
                    clicked_polygon = st_data['last_active_drawing']['geometry']
                    coordinates = clicked_polygon['coordinates'][0]
                    polygon = Polygon(coordinates)

                    clicked_info = gdf_station_info.loc[gdf_station_info.within(polygon)]
                    clicked_id_list   = clicked_info['NO'].to_list()
                    clicked_name_list = clicked_info['ID'].to_list()
                    clicked_elev_list = clicked_info['ELEV'].to_list()

                    df_all = pd.DataFrame()
                    date_list = pd.date_range(start=year+'-01-02', end=year+'-12-10')
                    df_all['date'] = date_list

                    df_all_sd = pd.DataFrame()
                    df_all_sd['date'] = date_list

                    # Get all station daily data
                    station_to_average = []
                    for i in range(len(clicked_id_list)):
                        clicked_id   = clicked_id_list[i]
                        clicked_name = clicked_name_list[i]
                        clicked_elev = clicked_elev_list[i]
 
                        # Read data
                        elev_rdrs, elevation_era5, df_station, station, df_station_sd, df_rdrs, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd = make_timeserie(year, clicked_id, clicked_name, clicked_elev, lapse_type, min_or_max, version, hour_range)

                        # Save station data in the full dataframe
                        mask = (df_station['date_from'] >= date_list[0]) & (df_station['date_from'] <= date_list[-1])
                        df_station    = df_station.loc[mask]
                        is_column_nan = df_station[min_or_max].isna().all()

                        if not is_column_nan:
                            station_to_average.append(clicked_id)
                            values = df_station[min_or_max]
                            df_all = pd.merge(df_all, values, left_on='date', right_index=True, how='left')
                            df_all.rename(columns={min_or_max: clicked_id+'-station'}, inplace=True)

                            mask = (df_station_sd['date'] >= date_list[0]) & (df_station_sd['date'] <= date_list[-1])
                            df_station_sd = df_station_sd.loc[mask]
                            df_station_sd.set_index('date', inplace=True)
                            values = df_station_sd['SD']
                            df_all_sd = pd.merge(df_all_sd, values, left_on='date', right_index=True, how='left')
                            df_all_sd.rename(columns={'SD': clicked_id+'-station'}, inplace=True)

                            # Save station data in the full dataframe
                            try:
                                mask = (df_era5['date_from'] >= date_list[0]) & (df_era5['date_from'] <= date_list[-1])
                                df_era5 = df_era5.loc[mask]
                                df_era5.set_index('date_from', inplace=True)
                                values = df_era5[min_or_max]
                                df_all = pd.merge(df_all, values, left_on='date', right_index=True, how='left')
                                df_all.rename(columns={min_or_max: clicked_id+'-ERA5'}, inplace=True)

                                df_era5_sd['date'] = df_era5_sd['date'].dt.date
                                mask = (df_era5_sd['date'] >= date_list[0]) & (df_era5_sd['date'] <= date_list[-1])
                                df_era5_sd = df_era5_sd.loc[mask]
                                values = df_era5_sd['SD'].to_list()
                                df_all_sd[clicked_id+'-ERA5'] = values

                            except:
                                continue

                            for v in version:
                                if v is not 'ERA5L':
                                    df_rdrs_temp = df_rdrs[v]
                                    mask = (df_rdrs_temp['date_from'] >= date_list[0]) & (df_rdrs_temp['date_from'] <= date_list[-1])
                                    df_rdrs_temp = df_rdrs_temp.loc[mask]
                                    df_rdrs_temp.set_index('date_from', inplace=True)
                                    values = df_rdrs_temp[min_or_max]
                                    df_all = pd.merge(df_all, values, left_on='date', right_index=True, how='left')
                                    df_all.rename(columns={min_or_max: clicked_id+'-'+v}, inplace=True)

                                    df_rdrs_sd_temp = df_rdrs_sd[v]
                                    df_rdrs_sd_temp = df_rdrs_sd_temp[df_rdrs_sd_temp['date'].dt.hour == 12]
                                    df_rdrs_sd_temp['date'] = df_rdrs_sd_temp['date'].dt.date
                                    mask = (df_rdrs_sd_temp['date'] >= date_list[0]) & (df_rdrs_sd_temp['date'] <= date_list[-1])
                                    df_rdrs_sd_temp = df_rdrs_sd_temp.loc[mask]
                                    df_rdrs_sd_temp.set_index('date', inplace=True)
                                    values = df_rdrs_sd_temp['SD'].to_list()
                                    df_all_sd[clicked_id+'-'+v] = values

                    # Create new columns for averages
                    columns_station = [f'{prefix}-station' for prefix in station_to_average if f'{prefix}-station' in df_all.columns]
                    columns_era5    = [f'{prefix}-ERA5'    for prefix in station_to_average if f'{prefix}-ERA5'    in df_all.columns]
                    df_all['average-station'] = df_all[columns_station].mean(axis=1)
                    df_all['average-era5']    = df_all[columns_era5].mean(axis=1)

                    columns_station = [f'{prefix}-station' for prefix in station_to_average if f'{prefix}-station' in df_all_sd.columns]
                    columns_era5    = [f'{prefix}-ERA5'    for prefix in station_to_average if f'{prefix}-ERA5'    in df_all_sd.columns]
                    df_all_sd['average-station'] = df_all_sd[columns_station].mean(axis=1)
                    df_all_sd['average-era5']    = df_all_sd[columns_era5].mean(axis=1)

                    for v in version:
                        columns_02P1 = [f'{prefix}-'+v    for prefix in station_to_average if f'{prefix}-'+v    in df_all.columns]
                        df_all['average-'+v] = df_all[columns_02P1].mean(axis=1)

                        columns_02P1 = [f'{prefix}-'+v    for prefix in station_to_average if f'{prefix}-'+v    in df_all_sd.columns]
                        df_all_sd['average-'+v]    = df_all_sd[columns_02P1].mean(axis=1)

                    print(df_all)
                    print(df_all_sd)

                    # Reformat data to be read by the function
                    df_station    = df_all[['date','average-station']]
                    df_station.rename(columns={'average-station': min_or_max, 'date': 'date_from'}, inplace=True)
                    df_station_sd = df_all_sd[['date','average-station']]
                    df_station_sd.rename(columns={'average-station': 'SD', 'date': 'date'}, inplace=True)
                    station       = True

                    df_rdrs_tt = {}
                    df_rdrs_sd = {}
                    for v in version:
                        df_rdrs_tt[v] = df_all[['date','average-'+v]]
                        df_rdrs_tt[v].rename(columns={'average-'+v: min_or_max, 'date': 'date_from'}, inplace=True)

                        df_rdrs_sd[v] = df_all_sd[['date','average-'+v]]
                        df_rdrs_sd[v].rename(columns={'average-'+v: 'SD', 'date': 'date'}, inplace=True)

                    df_era5 = df_all[['date','average-era5']]
                    df_era5.rename(columns={'average-era5': min_or_max, 'date': 'date_from'}, inplace=True)
                    df_era5_sd = df_all_sd[['date','average-era5']]
                    df_era5_sd.rename(columns={'average-era5': 'SD', 'date': 'date'}, inplace=True)
                    era5 = True
                    
                    fig = make_figure('region', version, min_or_max, df_station, station, df_station_sd, df_rdrs_tt, rdrs_tt, df_rdrs_sd, rdrs_sd, df_era5, era5, df_era5_sd)
                    st.write(fig)

                    # Metadata
                    st.subheader('Data about averaged timeserie:')

                    metadata = clicked_info[['ID','NO','LAT','LON','ELEV']]
                    metadata['in average'] = metadata['NO'].isin(station_to_average)

                    st.write(metadata)
                    #st.write(df_station_info[df_station_info['NO'] == station_to_average])

                except:
                    st.write('Click on a polygon')















