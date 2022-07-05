import numpy as np
import pandas as pd
import streamlit as st
#import datetime as dt
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
import glob
from streamlit_folium import folium_static, st_folium
import folium
from branca.colormap import linear, LinearColormap
#from backend import make_map, make_timeserie

matplotlib.use("agg")
_lock = RendererAgg.lock

# Compability between pandas versions and mpl
pd.plotting.register_matplotlib_converters()

# Wide streamlit page
st.set_page_config(layout="wide")

@st.cache(hash_funcs={folium.folium.Map: lambda _: None}, allow_output_mutation=True)
def make_map(df_station_info, field_to_color_by):
    main_map = folium.Map(location=(52, -121), zoom_start=5)
    colormap = linear.RdYlGn_07.scale(-6,0)
    colormap.add_to(main_map)

    for i in df_station_info.index:
        lat  = df_station_info['LAT'].loc[i]
        lon  = df_station_info['LON'].loc[i]
        elev = df_station_info['ELEV'].loc[i]
        name = df_station_info['ID'].loc[i]

        if np.isnan( df_station_info[field_to_color_by].loc[i] ):
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

def add_lapse_rate(lapse_type, date_debut, date_fin, elevation_station, elevation_rdrs):
    # Date array: dt = 1 day
    date1 = datetime(int(date_debut[0:4]),int(date_debut[5:7]),int(date_debut[8:10]))
    date2 = datetime(int(date_fin[0:4]),  int(date_fin[5:7]),  int(date_fin[8:10])) + timedelta(days=1)
    date_list = np.arange(date1, date2, timedelta(days=1)).astype(datetime)
    
    if lapse_type == 'none':
        lapse_rate = np.zeros_like(date_list)
    elif lapse_type == 'fixed':
        lapse_rate = np.zeros_like(date_list)
        lapse_rate = lapse_rate + 4.5*(elevation_rdrs - elevation_station)/1000
    elif lapse_type == 'Stahl':
        lapse_rate = []
        stahl = {
            '01': -3,
            '02': -4,
            '03': -7,
            '04': -8,
            '05': -8,
            '06': -8,
            '07': -8,
            '08': -7,
            '09': -7,
            '10': -6,
            '11': -4,
            '12': -4
        }
        for date in date_list:
            month = date.month
            if month < 10:
                month = '0'+str(month)
            else:
                month = str(month)
            lapse_rate.append( -1 * stahl[month]*(elevation_rdrs - elevation_station)/1000 )

    return lapse_rate

def make_timeserie(year, clicked_id, clicked_name, clicked_hourly, clicked_elev, lapse_type):
    # Dates
    date_debut = year+'-01-01'
    date_fin   = year+'-12-31'
 
    # Observations
    df_station = pd.read_pickle("data/"+clicked_id+"-station.pkl")

    df_station_sd = pd.DataFrame()
    if 'SD' in df_station.columns:
        df_station_sd['date'] = df_station['date']
        df_station_sd['SD']   = df_station['SD']
        mask = (df_station_sd['date'] > date_debut) & (df_station_sd['date'] <= date_fin)
        df_station_sd = df_station_sd.loc[mask]

    def func_station(val):
        minimum_val = df_station[val['date_from'] : val['date_to']]['TT'].min()
        maximum_val = df_station[val['date_from'] : val['date_to']]['TT'].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

    def func_station_daily(val):
        minimum_val = df_station[val['date_from'] : val['date_to']]['Tmin'].min()
        maximum_val = df_station[val['date_from'] : val['date_to']]['Tmax'].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

    if clicked_hourly:
        date_list = pd.date_range(start=date_debut, end=date_fin)
        df_temp = pd.DataFrame()
        df_temp['date_from'] = date_list
        df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

        df_station.set_index('date', inplace=True)
        df_station = pd.concat(list(df_temp.apply(func_station, axis=1)))

    elif not clicked_hourly:
        date_list = pd.date_range(start=date_debut, end=date_fin)
        df_temp = pd.DataFrame()
        df_temp['date_from'] = date_list
        df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

        df_station.set_index('date', inplace=True)
        df_station = pd.concat(list(df_temp.apply(func_station_daily, axis=1)))

    # RDRS
    df_rdrs = pd.read_pickle("data/"+clicked_id+"-RDRS.pkl")
    elevation = df_rdrs['elev'].loc[0]

    df_rdrs_sd = pd.DataFrame()
    if 'SD' in df_rdrs.columns:
        df_rdrs_sd['date'] = df_rdrs['date']
        df_rdrs_sd['SD']   = df_rdrs['SD']
        mask = (df_rdrs_sd['date'] > date_debut) & (df_rdrs_sd['date'] <= date_fin)
        df_rdrs_sd = df_rdrs_sd.loc[mask]

    def func_rdrs(val):
        minimum_val = df_rdrs[val['date_from'] : val['date_to']]['TT'].min()
        maximum_val = df_rdrs[val['date_from'] : val['date_to']]['TT'].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

    date_list = pd.date_range(start=date_debut, end=date_fin)
    df_temp = pd.DataFrame()
    df_temp['date_from'] = date_list
    df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

    df_rdrs.set_index('date', inplace=True)
    df_rdrs = pd.concat(list(df_temp.apply(func_rdrs, axis=1)))

    # Lapse rate
    lapse_rate = add_lapse_rate(lapse_type, date_debut, date_fin, clicked_elev, elevation)
    lapse_rate = np.array(lapse_rate)

    # Plot
    date = df_station['date_from'].to_list()
    temp_station_min = np.array(df_station['Tmin'].to_list()) 
    temp_station_max = np.array(df_station['Tmax'].to_list()) 
    temp_rdrs_min = df_rdrs['Tmin'].to_list()
    temp_rdrs_max = np.array(df_rdrs['Tmax'].to_list())

    biais = (temp_rdrs_max + lapse_rate) - temp_station_max
    print(biais)

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(date, temp_station_max)
    ax1.plot(date, (temp_rdrs_max + lapse_rate))
    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])

    if not df_rdrs_sd.empty:
        ax2 = ax1.twinx()
        ax2.plot(df_station_sd['date'], df_station_sd['SD'], '--')
        ax2.plot(df_rdrs_sd['date'],    df_rdrs_sd['SD'], '--')
        ax2.set_ylabel('Snow depth [cm]')
        ax2.set_ylim([-5,500])

    ax1.grid(True)

    plt.legend(['Tmax obs','Tmax RDRSv2.1', 'SD obs', 'SD RDRSv2.1'])
    plt.title('Tmax at '+clicked_name)

    return fig, elevation, biais


st.write('Hourly stations')

dataset = st.selectbox('Dataset',['ECCC network','BC archive'])

year = st.radio('Pick the year',['2017','2018'])

if dataset == 'ECCC network':
    df_station_info = pd.read_csv('data/station-biais-eccc.obs', delim_whitespace=True, skiprows=2)
elif dataset == 'BC archive':
    df_station_info = pd.read_csv('data/station-biais-canswe.obs', delim_whitespace=True, skiprows=2)
main_map = make_map(df_station_info, 'DATA.BIAIS_2017')

col1, col2, col3 = st.columns([0.7,0.3,1])

with col1:
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
    clicked_hourly = clicked_info['DATA.HOURLY'].to_list()[0]
    clicked_elev   = clicked_info['ELEV'].to_list()[0]
    if clicked_hourly == 1: clicked_hourly = True
    else:                   clicked_hourly = False

    with col3:
        st.header("Timeserie")
        lapse_type = st.radio('Lapse rate type',['none','fixed','Stahl'])
        #try:
        fig, elevation, biais = make_timeserie(year, clicked_id, clicked_name, clicked_hourly, clicked_elev, lapse_type)
        st.write(fig)
        #except:
        #    st.write("No data yet")

    with col2:
        st.header("Information")
        st.write("Latitude:", clicked_lat)
        st.write("Longitude:", clicked_lon)
        st.write("Station elevation:", clicked_elev)
        st.write("Model elevation:", elevation)

        biais_mean = np.nanmean(biais, dtype='float32')
        st.write("Biais sur la periode:", biais_mean)

