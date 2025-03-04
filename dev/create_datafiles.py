import os
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

import importlib.resources as importlib_resources
import zarr

import mdreg


def create_arr_molli_tiny():
    
    data = mdreg.fetch('MOLLI')
    fac = (1/16, 1/16, 1, 1)
    data['array'] = zoom(data['array'], zoom=fac, order=1)
    data['pixel_spacing'] = [d/fac[i] for i, d in enumerate(data['pixel_spacing'])]

    f = importlib_resources.files('mdreg.datafiles')
    save_path = str(f.joinpath('MOLLI_tiny.pkl'))
    with open(save_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def create_arr_molli_small():
    
    data = mdreg.fetch('MOLLI')
    fac = (1/4, 1/4, 1, 1)
    data['array'] = zoom(data['array'], zoom=fac, order=1)
    data['pixel_spacing'] = [d/fac[i] for i, d in enumerate(data['pixel_spacing'])]

    f = importlib_resources.files('mdreg.datafiles')
    save_path = str(f.joinpath('MOLLI_small.pkl'))
    with open(save_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def create_arr_vfa_tiny():
    
    data = mdreg.fetch('VFA')
    fac = (0.1, 0.1, 0.1, 1)
    data['array'] = zoom(data['array'], zoom=fac, order=1)
    data['spacing'] = [d/fac[i] for i, d in enumerate(data['spacing'])]

    f = importlib_resources.files('mdreg.datafiles')
    save_path = str(f.joinpath('VFA_tiny.pkl'))
    with open(save_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def create_arr_vfa_small():
    
    data = mdreg.fetch('VFA')
    fac = (0.25, 0.25, 0.25, 1)
    data['array'] = zoom(data['array'], zoom=fac, order=1)
    data['spacing'] = [d/fac[i] for i, d in enumerate(data['spacing'])]

    f = importlib_resources.files('mdreg.datafiles')
    save_path = str(f.joinpath('VFA_small.pkl'))
    with open(save_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def create_zarr_vfa_tiny():

    data = mdreg.fetch('VFA_tiny')
    array = data['array']   

    f = importlib_resources.files('mdreg.datafiles')
    store = str(f.joinpath('VFA_tiny.zarr'))
    zarray = zarr.create(
        shape=array.shape, dtype='float32', 
        chunks=array.shape[:2] + (1,1),
        store=store)
    zarray[:] = data['array']
    zarray.attrs['FA'] = list(data['FA'])
    zarray.attrs['spacing'] = list(data['spacing'])


def create_zarr_vfa_small():

    data = mdreg.fetch('VFA_small')
    array = data['array']   

    f = importlib_resources.files('mdreg.datafiles')
    store = str(f.joinpath('VFA_small.zarr'))
    zarray = zarr.create(
        shape=array.shape, dtype='float32', 
        chunks=array.shape[:2] + (1,1),
        store=store)
    zarray[:] = data['array']
    zarray.attrs['FA'] = list(data['FA'])
    zarray.attrs['spacing'] = list(data['spacing'])


def create_zarr_vfa():

    data = mdreg.fetch('VFA')
    array = data['array']   

    f = importlib_resources.files('mdreg.datafiles')
    store = str(f.joinpath('VFA.zarr'))
    zarray = zarr.create(
        shape=array.shape, dtype='float32', 
        chunks=array.shape[:2] + (1,1),
        store=store)
    zarray[:] = data['array']
    zarray.attrs['FA'] = list(data['FA'])
    zarray.attrs['spacing'] = list(data['spacing'])


def create_zarr_dce():

    f = importlib_resources.files('mdreg.datafiles')
    
    # Note: sourcre data not included in mdreg
    data = mdreg.fetch_zarr('disco')
    curves = pd.read_excel(str(f.joinpath('MEDCIC_02_B1.xlsx')))
    time = curves.time.values
    aif = curves.aorta.values
    spacing = [1.95, 1.95, 5.0]

    zarray = zarr.create(
        shape=data.shape, 
        dtype=data.dtype, 
        chunks=data.shape[:2] + (1,1),
        store=str(f.joinpath('DCE.zarr')),
    )
    zarray[:] = data
    zarray.attrs['spacing'] = spacing
    zarray.attrs['time'] = (time-time[0]).tolist()
    zarray.attrs['aif'] = aif.tolist()
    

def create_zarr_dce_small():

    f = importlib_resources.files('mdreg.datafiles')
    
    # Note: sourcre data not included in mdreg
    data = mdreg.fetch_zarr('disco')
    curves = pd.read_excel(str(f.joinpath('MEDCIC_02_B1.xlsx')))
    time = curves.time.values
    aif = curves.aorta.values
    spacing = [1.95, 1.95, 5.0]
    n0, n1 = 25, 60
    fac = (1/8.0, 1/8.0, 1/3.0, 1)
    arr = zoom(data[:,:,:,n0:n1], zoom=fac, order=1)
    arr = arr[:,:,1:,:]

    zarray = zarr.create(
        shape=arr.shape, 
        dtype=arr.dtype, 
        chunks=arr.shape[:2] + (1,1),
        store=str(f.joinpath('DCE_small.zarr')),
    )
    zarray[:] = arr
    zarray.attrs['spacing'] = [d/fac[i] for i, d in enumerate(spacing)]
    zarray.attrs['time'] = (time[n0:n1]-time[n0]).tolist()
    zarray.attrs['aif'] = aif[n0:n1].tolist()


def create_zarr_dce_tiny():

    f = importlib_resources.files('mdreg.datafiles')
    
    # Note: sourcre data not included in mdreg
    data = mdreg.fetch_zarr('disco')
    curves = pd.read_excel(str(f.joinpath('MEDCIC_02_B1.xlsx')))
    time = curves.time.values
    aif = curves.aorta.values
    spacing = [1.95, 1.95, 5.0]
    n0, n1 = 25, 35
    fac = (1/32.0, 1/32.0, 1/12.0, 1)
    arr = zoom(data[:,:,:,n0:n1], zoom=fac, order=1)
    arr = arr[:,:,1:,:]

    zarray = zarr.create(
        shape=arr.shape, 
        dtype=arr.dtype, 
        chunks=arr.shape[:2] + (1,1),
        store=str(f.joinpath('DCE_tiny.zarr')),
    )
    zarray[:] = arr
    zarray.attrs['spacing'] = [d/fac[i] for i, d in enumerate(spacing)]
    zarray.attrs['time'] = (time[n0:n1]-time[n0]).tolist()
    zarray.attrs['aif'] = aif[n0:n1].tolist()



if __name__ == '__main__':

    create_arr_molli_tiny()
#    create_arr_molli_small()
#    create_arr_vfa_tiny()
#    create_arr_vfa_small()
#    create_zarr_vfa_tiny()
#    create_zarr_vfa_small()
#    create_zarr_vfa()
#    create_zarr_dce()
#    create_zarr_dce_small()
#    create_zarr_dce_tiny()