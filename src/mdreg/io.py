import os
import shutil

import numpy as np
import zarr



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
        fit = np.zeros(signal.shape)
        par = np.zeros(signal.shape[:-1] + (npar,))
        return fit, par

    # zarrays in memory
    if path is None:
        fit = zarr.create(
            signal.shape, 
            dtype=signal.dtype, 
            chunks=signal.chunks,
            store=zarr.MemoryStore(),
        )
        fit[:] = 0
        par = zarr.create(
            signal.shape[:-1] + (npar, ), 
            dtype=signal.dtype, 
            chunks=signal.chunks[:-1] + (npar,),
            store=zarr.MemoryStore(),
        )
        par[:] = 0
        return fit, par
    
    # zarrays on disk
    os.makedirs(path, exist_ok=True)
    store_fit = os.path.join(path, 'fit.zarr')
    store_par = os.path.join(path, 'pars.zarr')
    # if os.path.exists(store_fit):
    #     # fit = zarr.open(store_fit, mode='w')
    #     # par = zarr.open(store_par, mode='w')
    #     fit = zarr.open_array(store_fit, mode='w')
    #     par = zarr.open_array(store_par, mode='w')
    # else:
    fit = zarr.create(
        signal.shape, 
        dtype=signal.dtype, 
        chunks=signal.chunks,
        store=store_fit,
        overwrite=True,
    )
    fit[:] = 0
    par = zarr.create(
        signal.shape[:-1] + (npar, ), 
        dtype=signal.dtype, 
        chunks=signal.chunks[:-1] + (npar,),
        store=store_par,
        overwrite=True,
    )
    par[:] = 0
    return fit, par


def _defo(array, path=None, force_2d=False, name='defo'):
   
    if array.ndim == 3: #2D
        dshape = array.shape[:3] + (2, ) 
    elif force_2d: #3D with 2D coreg
        dshape = array.shape[:4] + (2, ) 
    else: #3D
        dshape = array.shape[:4] + (3, ) 

    # Numpy arrays in memory
    if isinstance(array, np.ndarray):
        defo = np.zeros(dshape)
        if path is not None:
            np.save(os.path.join(path, name), defo)
        return defo
    
    # Zarrays in memory
    if path is None:
        defo = zarr.create(
            dshape, 
            dtype=array.dtype, 
            chunks=array.chunks + (dshape[-1], ),
            store=zarr.MemoryStore(),
        )
        defo[:] = 0
        return defo
    
    # Zarrays on disk
    os.makedirs(path, exist_ok=True)
    store_defo = os.path.join(path, name+'.zarr')
    defo = zarr.create(
        shape=dshape, 
        dtype=array.dtype, 
        chunks=array.chunks + (dshape[-1], ),
        store=store_defo,
        overwrite=True,
    )
    defo[:] = 0
    return defo


def _copy(array, path=None, name='copy'):

    # Numpy arrays in memory
    if isinstance(array, np.ndarray):
        copy = array.copy()
        if path is not None:
            np.save(os.path.join(path, name), copy)
        return copy
    
    # Zarrays in memory
    if path is None:
        copy = zarr.create(
            shape=array.shape, 
            dtype=array.dtype,
            chunks=array.chunks,
            store=zarr.MemoryStore(),
        )
        copy[:] = array
        return copy
    
    # Zarrays on disk
    os.makedirs(path, exist_ok=True)
    copy = zarr.create(
        shape=array.shape, 
        dtype=array.dtype, 
        chunks=array.chunks,
        store=os.path.join(path, name+'.zarr'),
        overwrite=True,
    )
    copy[:] = array
    return copy