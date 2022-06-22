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
from backend import make_map, make_timeserie

matplotlib.use("agg")
_lock = RendererAgg.lock

# Compability between pandas versions and mpl
pd.plotting.register_matplotlib_converters()

# Wide streamlit page
st.set_page_config(layout="wide")

st.write('Hourly stations')

year = st.radio('Pick the year',['2017','2018'])

df_station_info = pd.read_csv('/home/aul001/reanalyse/obs/station-hourly/station-info.obs', delim_whitespace=True, skiprows=2)
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

    with col3:
        st.header("Timeserie")
        try:
            fig, elevation, biais = make_timeserie(year, clicked_id, clicked_name)
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





