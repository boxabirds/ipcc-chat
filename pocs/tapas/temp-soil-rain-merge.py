# we have 3 data sources with values for given latlong values. But how
# much do they overlap? 
import netCDF4 as nc
import numpy as np
import pandas as pd

def load_lat_lon(filename):
    ds = nc.Dataset(filename)
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    ds.close()
    return pd.DataFrame({'Latitude': lat, 'Longitude': lon})

# Load latitude and longitude from each file
df_temp = load_lat_lon('SYR_Figure_SPM.2a_cmip6_TXx_change_at_1.5C.nc')
df_soil_temp = load_lat_lon('SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_1.5C.nc')
df_precip = load_lat_lon('SYR_Figure_SPM.2c_cmip6_Rx1day_change_at_1.5C.nc')

# Find common lat/lon pairs
common_coords = df_temp.merge(df_soil_temp, on=['Latitude', 'Longitude'], how='inner')
common_coords = common_coords.merge(df_precip, on=['Latitude', 'Longitude'], how='inner')

print(f'Number of common coordinate pairs: {len(common_coords)}')
print(common_coords.head())
