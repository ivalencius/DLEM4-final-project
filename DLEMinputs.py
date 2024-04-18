import xarray as xr
import cf_xarray as cfxr
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
from xclim import sdba
from dask.distributed import Client, LocalCluster
import dask_jobqueue
import sys

# Using CESM2LENS2 model run 1231 h1

DLEM_DIR = '/mmfs1/data/valencig/DLEM4-final-project/data/DLEM/'
MODEL_DIR = '/mmfs1/data/valencig/DLEM4-final-project/data/DLEM/CESM2LENS2/'

def get_client(basic=False):
    if basic:
        cluster = LocalCluster(
            n_workers=4,
            threads_per_worker=1,
        )
        return Client(cluster)
    # Create our NCAR Cluster - which uses PBSCluster under the hood
    num_jobs = 5

    cluster = dask_jobqueue.SLURMCluster(
        job_name='valencig_dask',
        cores=1,  # Total number of cores per job
        memory='10GB', # Total amount of memory per job
        processes=1, # Number of Python processes per job
        # interface='hsn0', # Network interface to use like eth0 or ib0
        # queue='main',
        walltime='01:00:00',
        # resource-spec: select=1:ncpus=128:mem=235GB
        # local_directory = '/glade/u/home/valencig/spilled/',
        # local_directory = '/glade/derecho/scratch/spilled/valencig/',
        log_directory = '/mmfs1/data/valencig/worker-logs/',
    )

    # Spin up workers
    cluster.scale(num_jobs)

    # Assign the cluster to our Client
    client = Client(cluster)

    # Block progress until workers have spawned
    client.wait_for_workers(num_jobs)
    return client

def create_interpolation_grid():
    """Create the interpolation grid for the DLEM model

    Returns:
        ncols, nrows, lons, lats, no_data_value: from mask_final.hdr
    """
    ncols = 693
    nrows = 292
    # Coordinates of the lower left corner of the lower left cell
    xllcornder = -124.78750228882+360
    yllcorner = 25.087501525879
    cellsize = 0.083333
    no_data_value = -9999
    # Create the lat lon grid for interpolation
    lons = np.linspace(xllcornder, xllcornder + ncols * cellsize, ncols)
    # Lats go from lower to upper so we need to reverse them
    lats = np.linspace(yllcorner + nrows * cellsize, yllcorner,  nrows)
    return ncols, nrows, lons, lats, no_data_value

def load_mask(n_lon, n_lat):
    """Loads the mask from the DLEM model.

    Args:
        n_lon (int): Number of longitudes.
        n_lat (int): Number of lat point.

    Returns:
        np.array: mask
    """
    mask = np.fromfile(DLEM_DIR+'mask_final.flt', dtype=np.float32).reshape([n_lat, n_lon])
    return mask

def load_data(model_name):
    """Loads data from the CESM2LEN2 climate model.

    Args:
        model_name (str): Name of variable in model.

    Returns:
        xr.DataArray: Data from the model.
    """
    files = glob(f'{MODEL_DIR}/{model_name}/*.nc')
    ds = xr.open_mfdataset(
        files,
        parallel=False,
        #preprocess=lambda x: x.sel(lat=slice(25, 55), lon=slice(-125+360, -60+360))
    )#.chunk({'lat': 10, 'lon':10})
    ds = ds.sel(lat=slice(25, 55), lon=slice(-125+360, -60+360), time=slice(None, '2015'))
    match model_name:
        case 'PRECC':
            # Convert from m/s to mm/day
            ds = ds * 86400000
            ds = ds.assign_attrs(units='mm/day')
        case 'TS', 'TSMX', 'TSMN':
            # Convert from K to Celcius
            ds = ds - 273.15
            ds = ds.assign_attrs(units='Celcius')
    # Coerce to np.float32 (8 bytes) -> equivalent to MATLAB double
    da_float32 = ds[model_name].astype(np.float32)
    return da_float32

def load_DLEM(model_name, lons, lats, dlem_name):
    units = dict(
        # Variable name: (DLEM name, model name)
        dswrf='W/m^2',      # Downward Shortwave Radiation [W/m^2]
        pr='mm/day',        # Precipitation [mm/day] <- originally in m/s
        tavg='Celcius',     # Average Temperature [Celcius] <- originally in K
        tmax='Celcius',     # Maximum Temperature [Celcius]
        tmin='Celcius',     # Minimum Temperature [Celcius]
    )
    files = sorted(glob(f'DLEM_DIR/Historical/{model_name}*.bin'))
    arrays = []
    for f in files:
        year = f[-8:-4]
        numpy = np.fromfile(f, dtype=np.float32).reshape([365, 292, 693])
        # Create the xarray dataset
        ds = xr.Dataset(
            {
                dlem_name: (['time', 'lat', 'lon'], numpy),
            },
            coords={
                'time': pd.date_range(f'{year}-01-01', periods=365),
                'lat': lats,
                'lon': lons
            },
            attrs={
                'units': units[model_name]
            }
        )
        arrays.append(ds)
    # Concatenate the arrays
    combined = xr.concat(arrays, dim='time')#.chunk({'lat':10, 'lon':10})
    return combined

def interpolate_data(da, mask, lons, lats, no_data_value):
    """Downsample the model data to 5km grid.

    Args:
        da (xr.DataArray): CESM2LENS2 data.
        mask (np.array): Mask for no data values.
        lons (np.array): Longitudes.
        lats (np.array): Latitudes.
        no_data_value (int): Value for no data.

    Returns:
        xr.DataArray: Interpolated and masked data.
    """
    new_data = da.cf.interp(longitude=lons, latitude=np.flip(lats), method='cubic')#.chunk({'lat':10, 'lon':10})
    # Add mask to dataset
    new_data['mask'] = (('lat', 'lon'), mask)
    # Set values of the mask to no_data_value
    applied_mask = new_data.where(mask!=no_data_value, other=no_data_value)
    # Remove mask
    drop_mask = applied_mask.drop_vars('mask')
    return drop_mask

def bias_correct(da, da_ref, dlem_name):
    ## USE DAY OF YEAR???????
    match dlem_name:
        case 'pr':
            # Local Intensity Scaling (LOCI) bias-adjustment.
            # USGS dry day criteria: https://earlywarning.usgs.gov/usraindry/rdreadme.php
            adj = sdba.adjustment.LOCI(da_ref, da, thresh='1 mm/day')
        case _:
            adj = sdba.adjustment.EmpiricalQuantileMapping(da_ref, da)
    # Now correct the data
    data_corrected = adj.adjust(da)
    return data_corrected

def save_data(data, dlem_name, n_lon, n_lat):
    """Write the data to binary files for the DLEM model.

    Args:
        data (xr.DataArray): CESM2LEN2 data.
        dlem_name (str): Save name.
        n_lon (int): Number of lon points.
        n_lat (int): Number of lat points.

    Returns:
        boolean: True.
    """
    # Save data to file
    data.to_netcdf(f'{DLEM_DIR}/{dlem_name}/{dlem_name}.nc')
    # Loop over years
    for year in range(data.cf['T'].dt.year.values.min(), data.cf['T'].dt.year.values.max()+1):
        # Extract array
        year_da = data.cf.sel(T=str(year)).astype(np.float32)
        print(year_da.max())
        # assert year_da.dtype == np.float32, "Data not 8 byte float"
        # assert year_da.cf['T'].size == 365, "Data not 365 days"
        # Reshape to 1D (finally load data)
        reshaped = year_da.values.reshape([n_lon*n_lat*365, 1])
        # Save to file
        reshaped.tofile(f'{DLEM_DIR}/{dlem_name}/{dlem_name}{year}.bin')
    return True

def process_DLEM_inputs():
    # Create a dask client
    client = get_client(basic=True)
    print(f'Dask client created: {client.dashboard_link}')
    # Step 1: Load the interpolation grid
    n_lon, n_lat, lons, lats, no_data_value = create_interpolation_grid()
    # Step 2: Load the data mask
    mask = load_mask(n_lon, n_lat)
    # Step 3: For each variable, load the data and interpolate it
    variables = dict(
        # Variable name: (DLEM name, model name)
        dswrf='FSA',    # Downward Shortwave Radiation [W/m^2]
        pr='PRECC',     # Precipitation [mm/day] <- originally in m/s
        tavg='TS',      # Average Temperature [Celcius] <- originally in K
        tmax='TSMX',    # Maximum Temperature [Celcius]
        tmin='TSMN',    # Minimum Temperature [Celcius]
    )
    print(f'Directories containing binaries will be created in `{DLEM_DIR}`')
    for dlem_name, model_name in tqdm(variables.items(), desc='Processing DLEM inputs', file=sys.stdout):
        if not os.path.exists(f'{DLEM_DIR}/{dlem_name}'):
            os.makedirs(f'{DLEM_DIR}/{dlem_name}')
        print('Loading data...')
        da = load_data(model_name)
        da = da.persist()
        # da_ref = load_DLEM(model_name, lons, lats, dlem_name)
        print('Interpolating data...')
        data = interpolate_data(da, mask, lons, lats, no_data_value)
        print('Computing data...')
        data = data.persist().compute()
        # Bias correct step!
        # data_corrected = bias_correct(data, da_ref)
        print('Saving data...')
        save_data(data, dlem_name, n_lon, n_lat)
    client.shutdown()
    return True

if __name__ == '__main__':
    process_DLEM_inputs()