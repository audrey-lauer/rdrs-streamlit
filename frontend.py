import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
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
from streamlit_folium import folium_static, st_folium
import folium
from branca.colormap import linear, LinearColormap
from backend import add_lapse_rate#, find_min_max
from shapely.geometry import Point, Polygon
import geopandas as gpd

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

def make_timeserie(year, clicked_id, clicked_name, clicked_hourly, clicked_elev, lapse_type):
    # Dates
    date_debut = year+'-01-02'
    date_fin   = year+'-11-06'
    date_list = pd.date_range(start=date_debut, end=date_fin)
 
    # Observations
    df_station_og = pd.read_pickle("data/"+clicked_id+"-station.pkl")

    df_station_sd = pd.DataFrame()
    if 'SD' in df_station_og.columns:
        df_station_sd['date'] = df_station_og['date']
        df_station_sd['SD']   = df_station_og['SD']
        mask = (df_station_sd['date'] > date_debut) & (df_station_sd['date'] <= date_fin)
        df_station_sd = df_station_sd.loc[mask]
        df_station_sd = df_station_sd[df_station_sd['SD'].notna()]

    print(df_station_og)
    df_station = find_min_max(df_station_og, date_list)
    print(df_station)

    # RDRS
    df_rdrs = pd.read_pickle("data/"+clicked_id+"-RDRS.pkl")
    df_rdrs = df_rdrs.drop_duplicates(subset='date')
    elevation_rdrs = df_rdrs['elev'].loc[0]

    try:
        df_rdrs_1stlevel = pd.read_pickle("data/"+clicked_id+"-RDRS-1st-level.pkl")
        df_rdrs_1stlevel = df_rdrs.drop_duplicates(subset='date')
    except:
        df_rdrs_1stlevel = pd.DataFrame()

    df_rdrs_sd = pd.DataFrame()
    if 'SD' in df_rdrs.columns:
        df_rdrs_sd['date'] = df_rdrs['date']
        df_rdrs_sd['SD']   = df_rdrs['SD']
        mask = (df_rdrs_sd['date'] > date_debut) & (df_rdrs_sd['date'] <= date_fin)
        df_rdrs_sd = df_rdrs_sd.loc[mask]

    df_rdrs = find_min_max(df_rdrs, date_list)

    if not df_rdrs_1stlevel.empty:
        df_rdrs_1stlevel = find_min_max(df_rdrs_1stlevel, date_list)

    # Lapse rate
    lapse_rate_rdrs = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_rdrs)
    lapse_rate_rdrs = np.array(lapse_rate_rdrs)

    # ERA5
    try:
        df_era5 = pd.read_pickle("data/"+clicked_id+"-ERA5.pkl")
        elevation_era5 = df_era5['elev'].iloc[0]

        df_era5_sd = pd.DataFrame()
        if 'SD' in df_era5.columns:
            df_era5_sd['date'] = df_era5['date']
            df_era5_sd['SD']   = df_era5['SD']
            mask = (df_era5_sd['date'] > date_debut) & (df_era5_sd['date'] <= date_fin)
            df_era5_sd = df_era5_sd.loc[mask]
            df_era5_sd = df_era5_sd[df_era5_sd['SD'].notna()]

        df_era5 = find_min_max(df_era5, date_list)

        # Lapse rate
        lapse_rate_era5 = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_era5)
        lapse_rate_era5 = np.array(lapse_rate_era5)

        era5 = True

    except:
        elevation_era5 = 0.
        era5 = False

    # GDRS
    try:
        df_gdrs = pd.read_pickle("data/"+clicked_id+"-GDRS.pkl")
        df_gdrs = df_gdrs.drop_duplicates(subset='date')

        df_gdrs_sd = pd.DataFrame()
        if 'SD' in df_gdrs.columns:
            df_gdrs_sd['date'] = df_gdrs['date']
            df_gdrs_sd['SD']   = df_gdrs['SD']
            mask = (df_gdrs_sd['date'] > date_debut) & (df_gdrs_sd['date'] <= date_fin)
            df_gdrs_sd = df_gdrs_sd.loc[mask]
            df_gdrs_sd = df_gdrs_sd[df_gdrs_sd['SD'].notna()]

        df_gdrs = find_min_max(df_gdrs, date_list)

        # Lapse rate
        #lapse_rate_gdrs = add_lapse_rate(lapse_type, date_list, clicked_elev, elevation_gdrs)
        #lapse_rate_gdrs = np.array(lapse_rate_gdrs)

        gdrs = True

    except:
        gdrs = False

    # Plot
    date = df_station['date_from'].to_list()
    temp_station_min = np.array(df_station['Tmin'].to_list()) 
    temp_station_max = np.array(df_station['Tmax'].to_list()) 
    temp_rdrs_min = np.array(df_rdrs['Tmin'].to_list())
    temp_rdrs_max = np.array(df_rdrs['Tmax'].to_list())

    if era5:
        temp_era5_min = np.array(df_era5['Tmin'].to_list())
        temp_era5_max = np.array(df_era5['Tmax'].to_list())

    if gdrs:
        temp_gdrs_min = np.array(df_gdrs['Tmin'].to_list())
        temp_gdrs_max = np.array(df_gdrs['Tmax'].to_list())

    #biais = (temp_rdrs_max + lapse_rate_rdrs) - temp_station_max
    biais = 0.

    fig, ax1 = plt.subplots(figsize=(10,5))

    tmax_obs  = ax1.plot(date, temp_station_max, 'k', label='Tmax obs')
    tmax_rdrs = ax1.plot(date, (temp_rdrs_max + lapse_rate_rdrs), 'b', label='Tmax RDRS')

    if era5: 
        tmax_era5 = ax1.plot(date, (temp_era5_max + lapse_rate_era5), 'g', label='Tmax ERA5')
        lns = tmax_obs + tmax_rdrs + tmax_era5
    else:
        lns = tmax_obs + tmax_rdrs

    if gdrs: 
        tmax_gdrs = ax1.plot(date, (temp_gdrs_max), 'm', label='Tmax GDRS')
        lns = lns + tmax_gdrs

    ax1.set_ylabel('Temperature [C]')
    ax1.set_ylim([-35,35])

    if not df_rdrs_1stlevel.empty:
        tmax_rdrs_1stlevel = ax1.plot(date, np.array(df_rdrs_1stlevel['Tmin'].to_list()), 'c', label='1st level RDRS')
        lns = lns + tmax_rdrs_1stlevel

    if not df_rdrs_sd.empty:
        ax2 = ax1.twinx()
        sd_obs  = ax2.plot(df_station_sd['date'], df_station_sd['SD'], '--k', label='SD obs')
        sd_rdrs = ax2.plot(df_rdrs_sd['date'],    df_rdrs_sd['SD'], '--b', label='SD RDRS')
        ax2.set_ylabel('Snow depth [cm]')
        ax2.set_ylim([-5,500])

        lns = lns + sd_obs + sd_rdrs

        if era5 and not df_era5_sd.empty:
            sd_era5 = ax2.plot(df_era5_sd['date'],    df_era5_sd['SD'], '--g', label='SD ERA5')

            lns = lns + sd_era5

        if gdrs and not df_gdrs_sd.empty:
            sd_gdrs = ax2.plot(df_gdrs_sd['date'],    df_gdrs_sd['SD'], '--m', label='SD GDRS')

            lns = lns + sd_gdrs

    ax1.grid(True)

    # added these three lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(1.08,1), borderaxespad=0)

    plt.title('Tmax at '+clicked_name)

    return fig, elevation_rdrs, elevation_era5, biais


st.write('Hourly stations')

dataset = st.selectbox('Dataset',['ECCC network','BC archive','RDRS - ERA5_land'])

if dataset == 'ECCC network' or dataset == 'BC archive':

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
    
        with col2:
            st.header("Parameters")
            st.write("Choose the parameters for timeserie")
    
            year = st.radio('Pick the year',['1996', '2017','2018'])
            lapse_type = st.radio('Lapse rate type',['none','fixed','Stahl'])
            min_or_max = st.radio('Tmin or Tmax?',['min','max'])
    
        with col3:
            st.header("Timeserie")
    
            fig, elevation_rdrs, elevation_era5, biais = make_timeserie(year, clicked_id, clicked_name, clicked_hourly, clicked_elev, lapse_type)
     
            df_elev = pd.DataFrame(index=['Station','RDRS','ERA5-land'])
            df_elev['Elevation (m)'] = [clicked_elev, elevation_rdrs, elevation_era5]
            st.dataframe(df_elev)
    
            st.write(fig)



elif dataset == 'RDRS - ERA5_land':
    col1, col2 = st.columns([0.5,0.5])

    with col1:
        year = st.radio('Pick the year',['2017','2018'])

        start_time, end_time = st.slider("Pick the date range",
                                         min_value=datetime(1999, 1, 1), 
                                         max_value=datetime(1999, 12, 31),
                                         value=(datetime(1999,1,1), datetime(1999,4,1)),
                                         format="MM/DD")

        start_time = start_time.replace(year=int(year))
        end_time   = end_time.replace(year=int(year))

    # Map
    with col2:
        ds = xr.open_dataset('data/map-'+year+'.nc')
        diff = ds['diff']
        diff = diff.sel(time=slice(start_time, end_time )) # Chose all hours in current date
        
        diff_average = diff.mean(dim="time").to_numpy()
        
        lon = ds['lon']
        lat = ds['lat']
        
        # Plot in map
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
        
        lat_min = 49
        lat_max = 60
        lon_min = 360-132
        lon_max = 360-120
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs.PlateCarree())
        
        resol = '10m'  # use data at this scale
        states = NaturalEarthFeature(category="cultural", scale=resol, facecolor="none", name="admin_1_states_provinces_shp")
        bodr = NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
        land = NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor="none")
        ocean = NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor="none")
        lakes = NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor="none")
        rivers = NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')
        
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
        cb.set_label('diff TT (C)')
 
        plt.title(start_time.strftime("%Y-%m-%d")+" to "+end_time.strftime("%Y-%m-%d") )
        
        st.pyplot(fig)

