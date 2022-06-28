import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
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
    colormap = linear.RdYlBu_08.scale(0,2000)
    colormap.add_to(main_map)

    for i in df_station_info.index:
        lat  = df_station_info['LAT'].loc[i]
        lon  = df_station_info['LON'].loc[i]
        elev = df_station_info['ELEV'].loc[i]
        name = df_station_info['ID'].loc[i]

        icon_color = colormap(elev)

        folium.CircleMarker(location=[lat, lon],
                    fill=True,
                    fill_color=icon_color,
                    color=None,
                    fill_opacity=0.7,
                    radius=5,
                    popup=name,
                    ).add_to(main_map)
    return main_map

def make_timeserie(year, clicked_id, clicked_name, clicked_hourly):
    #if clicked_id == '1060844':
    # Dates
    date_debut = year+'-01-01'
    date_fin   = year+'-12-31'

    # Observations
    df_station = pd.read_pickle("data/"+clicked_id+"-station.pkl")

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

    # Plot
    date = df_station['date_from'].to_list()
    temp_station_min = df_station['Tmin'] 
    temp_station_max = df_station['Tmax'] 
    temp_rdrs_min = df_rdrs['Tmin']
    temp_rdrs_max = df_rdrs['Tmax']
    biais = temp_rdrs_max - temp_station_max

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(date, temp_station_max)
    ax.plot(date, temp_rdrs_max)

    plt.ylabel('Temperature [C]')
    ax.set_ylim([-35,35])
    ax.grid(True)

    plt.legend(['obs','RDRSv2.1'])
    plt.title('Tmax at '+clicked_name)

    return fig, elevation, biais


st.write('Hourly stations')

dataset = st.selectbox('Dataset',['ECCC network','BC archive'])

year = st.radio('Pick the year',['2017','2018'])

if dataset == 'ECCC network':
    df_station_info = pd.read_csv('data/station-info-eccc.obs', delim_whitespace=True, skiprows=2)
elif dataset == 'BC archive':
    df_station_info = pd.read_csv('data/station-info-canswe.obs', delim_whitespace=True, skiprows=2)
main_map = make_map(df_station_info, 'ELEV')

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
    if clicked_hourly == 1: clicked_hourly = True
    else:                   clicked_hourly = False

    with col3:
        st.header("Timeserie")
        try:
            fig, elevation, biais = make_timeserie(year, clicked_id, clicked_name, clicked_hourly)
            st.write(fig)
        except:
            st.write("No data yet")

    with col2:
        st.header("Information")
        st.write("Latitude:", clicked_lat)
        st.write("Longitude:", clicked_lon)
        st.write("Station elevation:", clicked_info['ELEV'].to_list()[0])
        st.write("Model elevation:", elevation)
        st.write("Biais sur la periode:", np.average(biais))





