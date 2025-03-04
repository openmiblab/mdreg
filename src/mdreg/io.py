import os
import shutil

import numpy as np
import zarr
from zarr.storage import MemoryStore
import dask.array as da



def _remove(path, name):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)
    store = os.path.join(path, name + '.zarr')
    if os.path.exists(store):
        shutil.rmtree(store)


def _fit_models_init(signal, path, npar):

    # numpy arrays in memory
    if isinstance(signal, np.ndarray):
        fit_numpy = np.zeros(signal.shape)
        par_numpy = np.zeros(signal.shape[:-1] + (npar,))
        return fit_numpy, par_numpy

    if path is None:
        # zarrays in memory 
        store_fit = MemoryStore()
        store_par = MemoryStore()
    else:
        # zarrays on disk
        os.makedirs(path, exist_ok=True)
        store_fit = os.path.join(path, 'fit.zarr')
        store_par = os.path.join(path, 'pars.zarr')
    
    fit_dask_array = da.zeros(
        signal.shape, 
        dtype=signal.dtype, 
        chunks=signal.chunks, 
    )
    fit_dask_array.to_zarr(store_fit, overwrite=True)
    fit_zarray = zarr.open_array(store_fit, mode='a')

    par_dask_array = da.zeros(
        signal.shape[:-1] + (npar, ), 
        dtype=signal.dtype, 
        chunks=signal.chunks[:-1] + (npar,),
    )
    par_dask_array.to_zarr(store_par, overwrite=True)
    par_zarray = zarr.open_array(store_par, mode='a')
    
    return fit_zarray, par_zarray


def _defo(array, path=None, force_2d=False, name='defo'):
   
    if array.ndim == 3: #2D
        dshape = array.shape[:3] + (2, ) 
    elif force_2d: #3D with 2D coreg
        dshape = array.shape[:4] + (2, ) 
    else: #3D
        dshape = array.shape[:4] + (3, ) 

    # Numpy arrays in memory
    if isinstance(array, np.ndarray):
        defo_numpy = np.zeros(dshape)
        if path is not None:
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, name), defo_numpy)
        return defo_numpy
    
    if path is None:
        # Zarrays in memory
        store = MemoryStore()
    else:
        # Zarrays on disk
        os.makedirs(path, exist_ok=True)
        store = os.path.join(path, name+'.zarr')


    defo_dask_array = da.zeros(
        dshape, 
        dtype=array.dtype, 
        chunks=array.chunks + (dshape[-1], ), 
    )
    defo_dask_array.to_zarr(store, overwrite=True)
    defo_zarray = zarr.open_array(store, mode='a')

    return defo_zarray


def _copy(array, path=None, name='copy'):

    # Numpy arrays in memory
    if isinstance(array, np.ndarray):
        copy = array.copy()
        if path is not None:
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, name), copy)
        return copy
    
    if path is None:
        # Zarray in memory
        store = MemoryStore()
    else:
        # Zarray on disk
        os.makedirs(path, exist_ok=True)  
        store = os.path.join(path, name+'.zarr')  

    dask_array = da.from_zarr(array)
    dask_array.to_zarr(store, overwrite=True)
    copy_zarr = zarr.open_array(store, mode='a')
    return copy_zarr
    

if __name__ == '__main__':

    # Create a sample Zarr array
    zarr_array = zarr.create(shape=(10, 10), dtype='f4', chunks=(5, 5), store=MemoryStore())

    # Copy the Zarr array (in memory)
    copied_zarr = _copy(zarr_array)
    print(copied_zarr.shape)
    print(copied_zarr[0,0])
    copied_zarr[0,0] = -1
    print(copied_zarr[0,0])

