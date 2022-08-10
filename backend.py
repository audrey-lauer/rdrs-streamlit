import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# add_lapse_rate
# Input:  - lapse_type (str)
#         - date_list (list) 
#         - elevation_station (float)
#         - elevation_rdrs (float)
# Output: - lapse_rate (list): same length as date_list
def add_lapse_rate(lapse_type, date_list, elevation_station, elevation_rdrs):
    date_list = date_list.to_list()

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


def find_min_max(df, date_list):

    def func(val):
        minimum_val = df_copy[val['date_from'] : val['date_to']]['TT'].min()
        maximum_val = df_copy[val['date_from'] : val['date_to']]['TT'].max()
        return    pd.DataFrame({'date_from':[val['date_from']], 'date_to':[val['date_to']], 'Tmin': [minimum_val], 'Tmax': [maximum_val] })

    try:
        df_temp = pd.DataFrame()
        df_temp['date_from'] = date_list
        df_temp['date_to']   = date_list + pd.Timedelta(hours=23)

        df_copy = df.copy()
        df_copy.set_index('date', inplace=True)
        df_copy = pd.concat(list(df_temp.apply(func, axis=1)))

    except:
        df_copy = df.copy()
        mask = (df_copy['date'] >= date_list[0]) & (df_copy['date'] <= date_list[-1])
        df_copy = df_copy.loc[mask]
        df_copy['date_from'] = df_copy['date']

        df_copy.set_index('date', inplace=True)
    
    return df_copy


