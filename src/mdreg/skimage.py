from typing import Tuple, Union

from tqdm import tqdm
import numpy as np
import zarr
import dask
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp as skiwarp

from mdreg import io


def defaults():
    """Default parameters of the optical flow method

    Returns:
        dict: Default parameter values
    """
    return {
        'attachment':15,
        'tightness':0.3,
        'num_warp':5,
        'num_iter':10,
        'tol':1e-4,
        'prefilter':False,
    }

def coreg_series(
        moving: Union[np.ndarray, zarr.Array], 
        fixed: Union[np.ndarray, zarr.Array], 
        parallel=False, 
        progress_bar=True, 
        path=None, 
        name='coreg',
        **kwargs,
    ):
    """
    Coregister two series of 2D images or 3D volumes.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t). 
    fixed : numpy.ndarray | zarr.Array
        The fixed target image or volume, in the same dimensions as the 
        moving image. 
    parallel : bool
        Set to True to parallelize the computations. Defaults to True.
    progress_bar : bool
        Show a progress bar during the computation. This keyword is ignored 
        if parallel = True. Defaults to False.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.
    kwargs : dict
        Any keyword argument accepted by `skimage.registration.optical_flow_tvl1`. 

    Returns
    -------
    coreg : numpy.ndarray | zarr.Array
        Coregistered series with the same dimensions as the moving image. 
    defo : numpy.ndarray | zarr.Array
        The deformation field with the same dimensions as *moving*, and one 
        additional dimension for the components of the vector field. If 
        *moving* has dimensions (x,y,t) and (x,y,z,t), then the deformation 
        field will have dimensions (x,y,t,2) and (x,y,z,t,3), respectively. 
        The displacement vectors are measured in voxel units. 
    """
    if parallel:
        if progress_bar:
            raise ValueError(
                "A progress bar cannot be shown when parallel=True. "
                "Set parallel=False or progress_bar=False. "
            )
    coreg = io._copy(moving, path, name)
    defo = io._defo(moving, path, name=name+'_defo')

    if parallel:
        tasks = []
        for t in range(moving.shape[-1]): 
            task_t = dask.delayed(_coreg_t)(
                t, moving, fixed, coreg, defo, **kwargs,
            )
            tasks.append(task_t)
        dask.compute(*tasks)
    else:
        for t in tqdm(
                range(moving.shape[-1]), 
                desc='Coregistering series', 
                disable=not progress_bar,
            ): 
            _coreg_t(t, moving, fixed, coreg, defo, **kwargs)
    
    return coreg, defo


def _coreg_t(t, moving, fixed, deformed, deformation, **kwargs):
    deformed[...,t], deformation[...,t,:] = coreg(
        moving[...,t], fixed[...,t], **kwargs,
    )

def transform_series(
        moving, 
        defo,  
        path=None, 
        name='deform',
    ):
    """
    Transforms a series of images using a transformation produced by 
    `mdreg.skimage.coreg_series()`.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t).  
    defo : numpy.ndarray | zarr.Array
        The deformation field with the same dimensions as *moving*, and one 
        additional dimension for the components of the vector field. 
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.

    Returns
    -------
        numpy.ndarray: The transformed image.
    """
    deformed = io._copy(moving, path, name)
    for t in range(moving.shape[-1]):
        deformed[...,t] = transform(moving[...,t], defo[...,t,:])
    return deformed


def coreg(
        moving: np.ndarray, 
        fixed: np.ndarray, 
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Coregister two 2D images or 3D volumes.
    
    Parameters
    ----------
    moving : numpy.ndarray
        The moving image with dimensions (x,y) or (x,y,z). 
    fixed : numpy.ndarray
        The fixed target image with the same shape as the moving image. 
    kwargs : dict
        Any keyword argument accepted by `skimage.optical_flow_tvl1`. 
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image in the same shape as the moving image.
    defo : numpy.ndarray | zarr.Array
        The deformation field with the same dimensions as *moving*, and one 
        additional dimension for the components of the vector field. If 
        *moving* has dimensions (x,y) and (x,y,z), then the deformation 
        field will have dimensions (x,y,2) and (x,y,z,3), respectively. 
        The displacement vectors are measured in voxel units. To retrieve 
        displacements in physical units, the components defo[...,i] of the 
        deformation field need to be multiplied with the voxel dimensions. 
    """

    if moving.ndim == 2: 
        return _coreg_2d(moving, fixed, **kwargs)
    if moving.ndim == 3:
        return _coreg_3d(moving, fixed, **kwargs)


def _coreg_2d(moving, fixed, **kwargs):

    # Does not work with float or mixed type for some reason
    moving, fixed, a, b, dtype = _torange(moving, fixed)
    
    rc, cc = np.meshgrid( 
        np.arange(moving.shape[0]), 
        np.arange(moving.shape[1]),
        indexing='ij')
    row_coords = rc
    col_coords = cc

    v, u = optical_flow_tvl1(fixed, moving, **kwargs)
    new_coords = np.array([row_coords + v, col_coords + u])
    deformation_field = np.stack([v, u], axis=-1)
    warped_moving = skiwarp(moving, new_coords, mode='edge', 
                            preserve_range=True)
    if a is not None:
        # Scale back to original range and type
        warped_moving = warped_moving.astype(dtype)
        warped_moving = (warped_moving-b)/a
        
    return warped_moving, deformation_field


def _coreg_3d(moving, fixed, **kwargs):

    moving, fixed, a, b, dtype = _torange(moving, fixed)
    
    rc, cc, sc = np.meshgrid( 
        np.arange(moving.shape[0]), 
        np.arange(moving.shape[1]),
        np.arange(moving.shape[2]),
        indexing='ij')
    row_coords = rc
    col_coords = cc
    slice_coords = sc

    v, u, w = optical_flow_tvl1(fixed, moving, **kwargs)
    new_coords = np.array([row_coords + v, col_coords + u, slice_coords+w])
    deformation_field = np.stack([v, u, w], axis=-1)
    warped_moving = skiwarp(moving, new_coords, mode='edge', 
                            preserve_range=True)

    if a is not None:
        # Scale back to original range and type
        warped_moving = warped_moving.astype(dtype)
        warped_moving = (warped_moving-b)/a

    return warped_moving, deformation_field



def transform(moving, defo):
    """Transforms an image with a deformation field.

    Args:
        moving (numpy.ndarray): The input 2D or 3D image array.
        defo (numpy.ndarray): The deformation field. This array has 3
            dimensions for a 2D image array and 4 dimensions for a 3D image 
            array.

    Returns:
        numpy.ndarray: The transformed image.
    """

    if moving.ndim == 2:
        rc, cc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            indexing='ij',
        )
        row_coords = rc
        col_coords = cc

        v = defo[:,:,:,0]
        u = defo[:,:,:,1]

        coords = np.array([row_coords + v, col_coords + u])
        return skiwarp(moving, coords, mode='edge', preserve_range=True)
    
    if moving.ndim == 3:
        rc, cc, sc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            np.arange(moving.shape[2]),
            indexing='ij',
        )
        row_coords = rc
        col_coords = cc
        slice_coords = sc

        v = defo[:,:,:,0]
        u = defo[:,:,:,1]
        w = defo[:,:,:,2]

        coords = np.array([row_coords + v, col_coords + u, slice_coords+w])
        return skiwarp(moving, coords, mode='edge', preserve_range=True)


def _torange(moving, fixed):

    dtype = moving.dtype

    if dtype in [np.half, np.single, np.double, np.longdouble]:

        # Stay away from the boundaries
        i16 = np.iinfo(np.int16)
        imin = float(i16.min) + 16
        imax = float(i16.max) - 16

        # get scaling coefficients
        amin = np.amin([np.amin(moving), np.amin(fixed)])
        amax = np.amax([np.amax(moving), np.amax(fixed)])
        if amax == amin:
            a = 1
            b = - amin
        else:
            a = (imax-imin)/(amax-amin)
            b = - a * amin + imin

        # Scale to integer range
        moving = np.around(a*moving + b).astype(np.int16)
        fixed = np.around(a*fixed + b).astype(np.int16)

        return moving, fixed, a, b, dtype
    
    else:
    
        # Not clear why this is necessary but does not work otherwise
        moving = moving.astype(np.int16)
        fixed = fixed.astype(np.int16)

        return moving, fixed, None, None, None