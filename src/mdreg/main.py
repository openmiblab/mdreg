
import time
import numpy as np
from tqdm import tqdm
import dask.array as da

from mdreg import fit_models, elastix, skimage, ants, io

# TODO: test optional dependencies -> skimage default
# TODO: user guide introduction


def fit(moving,
        fit_pixel = None,
        fit_coreg = None,
        fit_image = None,
        tol = 1e-6,    
        maxit = 5,
        verbose = 0,
        force_2d = False,
        path = None, 
    ):
    """
    Remove motion from a series of 2D- or 3D images.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The series of images to be corrected, with dimensions (x,y,t) or (x,y,z,t). 
    fit_pixel : dict, optional
        A dictionary defining a single-pixel signal model. The possible items 
        in the dictionary are the keywords of the function `mdreg.fit_pixel`. 
        For a slice-by-slice computation (4D array with force_2d=True), 
        *fit_pixel* can be a list of dictionaries, one for each slice. 
        The default is None.
    fit_coreg : dict, optional
        The parameters for coregistering the images. *fit_coreg* has one 
        required item 'package' with possible values 'skimage' (default), 
        'elastix' and 'ants'. The other parameters are the possible keywords 
        of the *coreg_series* function of the package specified. 
    fit_image : dict or list, optional
        A dictionary defining the function to fit the signal data, and its 
        parameter values. This argument is ignored if *fit_pixel* is already 
        provided. *fit_image* has one required key 'func' that specifies the 
        fit function to use. The other entries are the keyword arguments of 
        this fit function. 
        The fit function can be one of the functions built in to mdreg, or a 
        custom made function. A valid fit function *must* take a signal array 
        as argument, and return two variables: an array with the same shape 
        containing the fit to the model, and a second variable that contains 
        the fitted parameters. 
        For a slice-by-slice computation (4D array with force_2d=True), 
        *fit_image* can be a list of dictionaries, one for each slice. 
        If *fit_image* is not provided, a constant model is used. 
    tol : float, optional
        Stopping criterion for the iteration. The iteration stops if the 
        largest difference between new and old coregistered series in any 
        pixel at any time point is less than *tol* of the largest value. 
        The default is 1e-6.
    maxit : int, optional
        The maximum number of iterations. The default is 0.
    verbose : int, optional
        The level of feedback to provide to the user. 0: no feedback; 1: text 
        output only; 2: text output and progress bars. The default is 2.
    force_2d : bool, optional
        By default, a 3-dimensional moving array will be coregistered with a 
        3-dimensional deformation field. To perform slice-by-slice 
        2-dimensional registration instead, set *force_2d* to True. This 
        keyword is ignored when the arrays are 2-dimensional. The 
        default is False.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.

    Returns
    -------
    coreg : numpy.ndarray | zarr.Array
        The coregistered images with the same dimensions as *moving*.
    fit : numpy.ndarray | zarr.Array
        The fitted signal model with the same dimensions as *arr*.
    transfo : numpy.ndarray | zarr.Array | list
        The parameters of the transformation deforming the moving image to the 
        coregistered image. With skimage, this is the deformation field with 
        the same dimensions as *moving*, and one additional dimension for the 
        components of the vector field. With elastix this is an array of 
        parameter objects and with ants this is an array of files with 
        transform parameters. Note when force_2d = True these are 2-dimensional 
        arrays with one transform per slice and per time point.
    pars : numpy.ndarray | zarr.Array
        The parameters of the fitted signal model with dimensions (x,y,n) or 
        (x,y,z,n), where n is the number of free parameters of the signal 
        model.
 
    """

    # Set defaults in fit_coreg
    if fit_coreg is None:
        fit_coreg = {'package': 'skimage'}
    if 'package' not in fit_coreg:
        fit_coreg['package'] = 'skimage'
    if 'progress_bar' not in fit_coreg:
        fit_coreg['progress_bar'] = verbose>1
    if 'name' not in fit_coreg:
        fit_coreg['name'] = 'coreg'

    # 2D slice-by-slice coregistration
    if moving.ndim==4:
        if force_2d:
            return  _fit_force_2d(
               moving, fit_image, fit_coreg, fit_pixel, tol, maxit, 
               verbose, path, 
            )
        
    # Set defaults for fit_image  
    if fit_image is None:
        fit_image = {'func': fit_models.fit_constant}

    # Check inputs
    if not isinstance(fit_image, dict):
        raise ValueError('The fit_image argument must be a dictionary.')

    # Set paths    
    _set_path(fit_coreg, path)
    _set_path(fit_image, path)
    _set_path(fit_pixel, path)

    # Compute
    converged = False
    it = 1
    start = time.time()

    print('Initializing..')
    coreg = io._copy(moving, path, 'coreg')

    while not converged: 

        startit = time.time()

        # Fit signal model
        if verbose > 0:
            print(f'Iteration {it}: fitting signal model')
        if fit_pixel is not None:
            fit, pars = fit_models.fit_pixels(coreg, **fit_pixel)
        else:
            kwargs = {i:fit_image[i] for i in fit_image if i!='func'}
            fit, pars = fit_image['func'](coreg, **kwargs)

        # Fit deformation
        if verbose > 0:
            print(f'Fitting deformation field (iteration {it})')
        coreg_curr = io._copy(coreg, path, 'tmp')
        vals = _coreg_series(moving, fit, **fit_coreg)
        coreg, transfo = vals[:2]

        # Check convergence
        converged = _diff(coreg, coreg_curr) < tol
        
        if verbose > 0:
            print(f'Calculation time for iteration {it}: '
                  f'{(time.time()-startit)/60} min')  

        if it == maxit: 
            break

        it += 1 

    if verbose > 0:
        print(f'Total calculation time: {(time.time()-start)/60} min')

    io._remove(path, 'tmp')
    if len(vals) > 2: # optional return value
        defo = vals[2]
        return coreg, fit, transfo, pars, defo
    else:
        return coreg, fit, transfo, pars


def _fit_force_2d(
        moving, fit_image, fit_coreg, fit_pixel, tol, maxit, verbose, 
        path,
    ):

    # Required outputs
    coreg = io._copy(moving, path, 'coreg')
    if fit_coreg['package'] == 'skimage':
        transfo = io._defo(
            moving, 
            path, 
            force_2d=True, 
            name=fit_coreg['name']+'_defo',
        )
    else:
        transfo = np.empty(moving.shape[-2:], dtype=object)

    # Optional outputs
    defo = None
    if 'return_deformation' in fit_coreg:
        if fit_coreg['return_deformation']:
            defo = io._defo(
                moving, 
                path, 
                force_2d=True, 
                name=fit_coreg['name']+'_defo',
            )

    for k in tqdm(
            range(moving.shape[2]), 
            desc='Fitting slice', 
            disable=verbose<2,
        ):
        if verbose == 1:
            print(f'Fitting slice {k+1} / {moving.shape[2]}')

        if fit_image is None:
            fit_image_k = None
        elif isinstance(fit_image, dict):
            fit_image_k = fit_image
        else:
            fit_image_k = fit_image[k]

        if fit_pixel is None:
            fit_pixel_k = None
        elif isinstance(fit_pixel, dict):
            fit_pixel_k = fit_pixel
        else:
            fit_pixel_k = fit_pixel[k]

        vals = fit(
            moving[:,:,k,:],
            fit_pixel = fit_pixel_k,
            fit_image = fit_image_k,
            fit_coreg = fit_coreg,
            tol = tol,
            maxit = maxit,
            verbose = verbose,
        )
        coreg[:,:,k,:], fit_k, transfo_k, pars_k = vals[:4]
        if k == 0:
            fit_arr, pars = io._fit_models_init(moving, path, pars_k.shape[-1])              
        if fit_coreg['package'] == 'skimage':
            transfo[:,:,k,:,:] = transfo_k
        else:
            transfo[k,:] = transfo_k
        fit_arr[:,:,k,:] = fit_k
        pars[:,:,k,:] = pars_k
        if defo is not None:
            defo[:,:,k,:,:] = vals[4]
    if defo is None:
        return coreg, fit_arr, transfo, pars
    else:
        return coreg, fit_arr, transfo, pars, defo



def _set_path(dct, path):
    if dct is None:
        return
    if path is None:
        return
    if 'path' in dct:
        if path != dct['path']:
            raise ValueError("Two different paths are provided.")
    else:
        dct['path'] = path
        

def _diff(coreg, coreg_curr):
    if isinstance(coreg, np.ndarray):
        corr = np.max(np.abs(coreg-coreg_curr))/np.max(np.abs(coreg_curr))
    else:
        coreg = da.from_zarr(coreg) 
        coreg_curr = da.from_zarr(coreg_curr)    
        corr = da.max(da.abs(coreg-coreg_curr))/da.max(da.abs(coreg_curr))
        corr.compute()
    return corr


def _coreg_series(moving, fit, package='elastix', **fit_coreg):

    if package == 'elastix':
        fit_coreg = _set_mdreg_elastix_defaults(fit_coreg)
        return elastix.coreg_series(moving, fit, **fit_coreg)
    
    elif package == 'skimage':
        return skimage.coreg_series(moving, fit, **fit_coreg)
    
    elif package == 'ants':
        return ants.coreg_series(moving, fit, **fit_coreg)
    
    else:
        raise NotImplementedError(
            'This coregistration package is not implemented')
    

def _set_mdreg_elastix_defaults(params):

    if "WriteResultImage" not in params:
        params["WriteResultImage"] = "false"
    if "WriteDeformationField" not in params:
        params["WriteDeformationField"] = "false"
    if "ResultImagePixelType" not in params:
        params["ResultImagePixelType"] = "float"

    # # Removing this for v0.4.2 as results appear to be worse
    # if 'Metric' not in params:
    #     params["Metric"] = "AdvancedMeanSquares"

    # # Settings pre v0.4.0 - unclear why - removed for now
    # if "FinalGridSpacingInPhysicalUnits" not in params:
    #     params["FinalGridSpacingInPhysicalUnits"] = "50.0"
    # if "AutomaticParameterEstimation" not in params:
    #     params["AutomaticParameterEstimation"] = "true"
    # if "ASGDParameterEstimationMethod" not in params:
    #     params["ASGDParameterEstimationMethod"] = "Original"
    # if "MaximumStepLength" not in params:
    #     params["MaximumStepLength"] = "1.0"
    # if "CheckNumberOfSamples" not in params:
    #     params["CheckNumberOfSamples"] = "true"
    # if "RandomCoordinate" not in params:
    #     params["ImageSampler"] = "RandomCoordinate"

    return params

    

