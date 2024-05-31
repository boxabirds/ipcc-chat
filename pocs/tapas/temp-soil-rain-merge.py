
import netCDF4 as nc
import numpy as np
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
import time
import sqlite3

def connect_to_geocache() -> sqlite3.Connection:
    connection = sqlite3.connect('geocode_cache.db')
    connection.execute('''
        CREATE TABLE IF NOT EXISTS geocode_cache (
            latitude REAL,
            longitude REAL,
            full_address TEXT,
            PRIMARY KEY (latitude, longitude)
        )
    ''')
    connection.commit()
    return connection

def reverse_geocode(lat, lon, conn: sqlite3.Connection):
    # Check cache first
    result = conn.execute('SELECT full_address FROM geocode_cache WHERE latitude=? AND longitude=?', (lat, lon)).fetchone()
    # we record when there is no address to avoid looking it up repeatedly
    # and there are a lot of them because bunches of addresses are in the ocean
    if result:
        return result[0] if result[0] != "N/A" else None
    
    # we have a new address that's not in the cache
    # so look it up and cache it -- even if the is no location found we cache that too 
    else:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        time.sleep(1)
        address = location.address if location else "N/A"
        # Insert new entry into the database
        conn.execute('INSERT INTO geocode_cache (latitude, longitude, full_address) VALUES (?, ?, ?)', (lat, lon, address))
        conn.commit()
        if address != "N/A":
            print(f"Geocoded: {address}")
        else:
            return address

def load_lat_lon_data(filename, variable_name, alias):
    # Open the .nc file
    ds = nc.Dataset(filename)
    # Extract lon and lat
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    # Extract the variable data
    data = ds.variables[variable_name][:]
    
    # Generate meshgrid for lon and lat to align with the 2D data variable
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Flatten the arrays for DataFrame creation
    df = pd.DataFrame({
        'Longitude': lon_grid.flatten(),
        'Latitude': lat_grid.flatten(),
        alias: data.flatten()  # Use alias for the column name
    })
    ds.close()
    return df

def merge_metrics(temp_value, c:sqlite3.Cursor, conn:sqlite3.Connection):
    # Format filenames based on the temperature value
    temp_filename = f'SYR_Figure_SPM.2a_cmip6_TXx_change_at_{temp_value}C.nc'
    soil_temp_filename = f'SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_{temp_value}C.nc'
    precip_filename = f'SYR_Figure_SPM.2c_cmip6_Rx1day_change_at_{temp_value}C.nc'

    # Load data with aliases
    df_temp = load_lat_lon_data(temp_filename, 'TXx', 'Average hottest day temperature change')
    df_soil_temp = load_lat_lon_data(soil_temp_filename, 'mrso', 'Annual mean total column soil moisture change')
    df_precip = load_lat_lon_data(precip_filename, 'Rx1day', 'Annual wettest-day precipitation change')

    # Report the number of rows
    print(f"Temperature {temp_value}C: Number of rows in temperature data (TXx): {len(df_temp)}")
    print(f"Temperature {temp_value}C: Number of rows in soil moisture data (mrso): {len(df_soil_temp)}")
    print(f"Temperature {temp_value}C: Number of rows in precipitation data (Rx1day): {len(df_precip)}")

    # Merge the dataframes on 'Longitude' and 'Latitude'
    merged_df = pd.merge(df_temp, df_precip, on=['Longitude', 'Latitude'], how='inner')
    merged_df = pd.merge(merged_df, df_soil_temp, on=['Longitude', 'Latitude'], how='left')

    # Check that every lat/lon entry has TXx and Rx1day values
    assert merged_df['Average hottest day temperature change'].notna().all() and merged_df['Annual wettest-day precipitation change'].notna().all(), "Some entries are missing TXx or Rx1day values."

    ## now add address to every entry in merged_df by calling reverse_geocode
    merged_df['Address'] = merged_df.apply(
        lambda row: reverse_geocode(
            row['Latitude'], 
            row['Longitude'], 
            c, conn), 
        axis=1
    )

    # Save the merged DataFrame to an HDF5 file
    hdf5_filename = f'SYR_Figure_SPM2_at_{temp_value}C.h5'
    merged_df.to_hdf(hdf5_filename, key='data', mode='w')

    print(f"Merged table has been saved to '{hdf5_filename}'.")

def load_and_label_data(temp_value):
    # Load the data from the HDF5 file
    hdf5_filename = f'SYR_Figure_SPM2_at_{temp_value}C.h5'
    df = pd.read_hdf(hdf5_filename, 'data')
    
    # Add a new column for the temperature value
    df['Average Global Temperature Increase'] = temp_value
    return df


geolocator = Nominatim(user_agent="ipcc chatbot data processor julian.harris@gmail.com")
conn = connect_to_geocache()

# temp, soil, and precip data are in separate tables. 
# merge them for each global average temperature model
merge_metrics("1.5", c, conn)
merge_metrics("2.0", c, conn)
merge_metrics("3.0", c, conn)
merge_metrics("4.0", c, conn)

# Load and label data for each temperature
df_1_5 = load_and_label_data("1.5")
df_2_0 = load_and_label_data("2.0")
df_3_0 = load_and_label_data("3.0")
df_4_0 = load_and_label_data("4.0")

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat([df_1_5, df_2_0, df_3_0, df_4_0])

# Save the final merged DataFrame to an HDF5 file
final_hdf5_filename = 'SYR_Figure_SPM2.h5'
final_df.to_hdf(final_hdf5_filename, key='data', mode='w')

print(f"Final merged table has been saved to '{final_hdf5_filename}'.")

# confirm it worked
final_df = pd.read_hdf(final_hdf5_filename, 'data')

# Print the number of rows in the DataFrame
print(f"Number of rows in the DataFrame: {len(final_df)}")

# Print the schema of the DataFrame
print("Schema of the DataFrame:")
print(final_df.dtypes)

# Display the first 5 rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(final_df.head())

conn.close()

# merged_df = pd.merge(df_temp, df_soil_temp, on=['Longitude', 'Latitude'], how='outer')
# merged_df = pd.merge(merged_df, df_precip, on=['Longitude', 'Latitude'], how='outer')

# # Count the total number of rows with 'mrso' values
# total_mrso_rows = merged_df['mrso'].notna().sum()

# # Count the number of rows where 'mrso' is not null but both 'TXx' and 'Rx1day' are null
# mrso_only_rows = merged_df[(merged_df['mrso'].notna()) & (merged_df['TXx'].isna()) & (merged_df['Rx1day'].isna())].shape[0]

# print(f"Total number of rows with 'mrso' values: {total_mrso_rows}")
# print(f"Number of rows with 'mrso' values but without 'TXx' and 'Rx1day' values: {mrso_only_rows}")

# # Filter rows where all three variables are non-null
# complete_cases = merged_df.dropna(subset=['TXx', 'mrso', 'Rx1day'])

# # Count the number of unique locations with complete data
# unique_complete_locations = len(complete_cases)

# print(f"Number of unique locations with complete data for all variables: {unique_complete_locations}")

# merged_df2 = pd.merge(df_temp, df_soil_temp, on=['Longitude', 'Latitude'], how='inner')
# merged_df2 = pd.merge(merged_df2, df_precip, on=['Longitude', 'Latitude'], how='inner')

# The resulting DataFrame, merged_df, now contains only rows where the latitude and longitude
# have non-null values in all three original DataFrames.
# print(f"Number of unique locations with data in all three variables: {len(merged_df2)}")

# # trying to pin down the presence of NaNs
# df_temp.reset_index(drop=True, inplace=True)
# df_soil_temp.reset_index(drop=True, inplace=True)
# df_precip.reset_index(drop=True, inplace=True)


# # Find common lat/lon pairs
# common_coords = df_temp.merge(df_soil_temp, on=['Longitude', 'Latitude'], how='inner')
# common_coords = common_coords.merge(df_precip, on=['Longitude', 'Latitude'], how='inner')

# print(f'Number of common coordinate pairs: {len(common_coords)}')
# print(common_coords.head())

# def inspect_data_sparsity(filename, variable_name):
#     ds = nc.Dataset(filename)
#     data = ds.variables[variable_name][:]
#     ds.close()
#     print(f"Data coverage for {variable_name}: {np.isnan(data).sum()} NaNs out of {data.size} total elements.")

# inspect_data_sparsity('SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_1.5C.nc', 'mrso')

import matplotlib.pyplot as plt

# def plot_data_distribution(filename, variable_name):
#     ds = nc.Dataset(filename)
#     data = ds.variables[variable_name][:]
#     lon = ds.variables['lon'][:]
#     lat = ds.variables['lat'][:]
#     lon_grid, lat_grid = np.meshgrid(lon, lat)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(lon_grid.flatten(), lat_grid.flatten(), c=data.flatten(), cmap='viridis', label=f"{variable_name} Data Points")
#     plt.colorbar(label=variable_name)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title(f'Data Distribution for {variable_name}')
#     plt.legend()
#     plt.show()
#     ds.close()

#plot_data_distribution('SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_1.5C.nc', 'mrso')

# def plot_data_distribution(filename, variable_name, ax, cmap, alpha=0.5):
#     ds = nc.Dataset(filename)
#     data = ds.variables[variable_name][:]
#     lon = ds.variables['lon'][:]
#     lat = ds.variables['lat'][:]
#     lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')
    
#     # Plotting
#     sc = ax.scatter(lon_grid.flatten(), lat_grid.flatten(), c=data.flatten(), cmap=cmap, alpha=alpha, label=f"{variable_name}", s=5)
#     plt.colorbar(sc, ax=ax, label=f"{variable_name} values")

# # Create figure and axes
# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot each dataset
# plot_data_distribution('SYR_Figure_SPM.2a_cmip6_TXx_change_at_1.5C.nc', 'TXx', ax, cmap='Reds', alpha=0.6)
# plot_data_distribution('SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_1.5C.nc', 'mrso', ax, cmap='Greens', alpha=0.6)
# plot_data_distribution('SYR_Figure_SPM.2c_cmip6_Rx1day_change_at_1.5C.nc', 'Rx1day', ax, cmap='Blues', alpha=0.6)

# ax.set_title('Data Distribution Overlap')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.legend()

# plt.show()


# def plot_coverage_map(filename, variable_name):
#     ds = nc.Dataset(filename)
#     data = ds.variables[variable_name][:]  # Assume 2D data array [lat, lon]
#     lon = ds.variables['lon'][:]
#     lat = ds.variables['lat'][:]
    
#     # Creating meshgrid with 'ij' indexing for proper alignment
#     lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')
    
#     # Check if data shape matches the grid shape
#     print("Data shape:", data.shape)
#     print("Lon grid shape:", lon_grid.shape)
#     print("Lat grid shape:", lat_grid.shape)

#     # Create a mask where data is not NaN
#     data_mask = np.isfinite(data)

#     plt.figure(figsize=(10, 5))
#     plt.title(f'Coverage Map for {variable_name}')
#     plt.scatter(lon_grid.flatten(), lat_grid.flatten(), c=data_mask.flatten(), s=1, alpha=0.5)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.grid(True)
#     plt.show()

#     ds.close()
# Plot coverage for each variable
# plot_coverage_map('SYR_Figure_SPM.2a_cmip6_TXx_change_at_1.5C.nc', 'TXx')
# plot_coverage_map('SYR_Figure_SPM.2b_cmip6_SM_tot_change_at_1.5C.nc', 'mrso')
# plot_coverage_map('SYR_Figure_SPM.2c_cmip6_Rx1day_change_at_1.5C.nc', 'Rx1day')