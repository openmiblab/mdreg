import __main__
import os
import warnings
from typing import Tuple, Union

from tqdm import tqdm
import numpy as np
import zarr
import dask

from mdreg import io

try:
    import itk
except:
    not_installed = True
else:
    not_installed = False



def defaults(method='bspline'):
    """The default elastix parameters

    Args:
        method (str, optional): Registration method. Options are 'bspline', 
          'affine', 'rigid', 'translation'. Defaults to 'bspline'.

    Returns:
        dict: Default parameters for the given method.
    """
    
    if method == 'bspline':
        return BSPLINE
    if method == 'affine':
        return AFFINE
    if method == 'rigid':
        return RIGID
    if method == 'translation':
        return TRANSLATION
    

def coreg_series(
        moving: Union[np.ndarray, zarr.Array], 
        fixed: Union[np.ndarray, zarr.Array], 
        parallel=True,
        progress_bar=False, 
        path=None, 
        name='coreg',
        return_deformation=False,
        spacing=1.0, 
        method='bspline', 
        **params,
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
    progress_bar: bool
        Display a progress bar during coregistration. This is ignored if 
        parallel is True. Defaults to False.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.
    return_deformation : bool
        If set to True, return the deformation field as a third return value.
    spacing: array-like
        Pixel spacing in mm. This can be a single scalar if all dimensions 
        are equal, or an array of 2 elements (for 2D data) or 3 elements (
        for 3D data). Defaults to 1. 
    method : str
        Deformation method to use. Options are 'bspline', 'affine', 'rigid' 
        or 'translation'. Default is 'bspline'.
    params : dict
        Use keyword arguments to overrule any of the default parameters in 
        the elastix template for the chosen method. The default parameters 
        can be found by printing mdreg.elastix.defaults().

    Returns
    -------
    coreg : numpy.ndarray | zarr.Array
        Coregistered series with the same dimensions as the moving image. 
    transfo : list
        List of itk.elastixParameterObject with one transform per image 
        in the series. The individuals transforms can be examined with 
        print(transfo[k]).
    """
    if not_installed:
        raise ImportError(
            "Coregistration with elastix is optional - please install mdreg as "
            "pip install mdreg[elastix] if you want to use these features."
        )
    
    if moving.shape != fixed.shape:
        raise ValueError('Moving and fixed arrays must have the '
                         'same shape.')

    coreg = io._copy(moving, path, name)
    transfo = np.empty(moving.shape[-1], dtype=object)

    if return_deformation:
        defo = io._defo(moving, path, name=name+'_defo')
    else:
        defo = None

    # This is a very slow step so needs to be done outside the loop
    if progress_bar:
        print('Building elastix parameter object..')
    p_obj = _params_obj(method, **params) 

    if parallel:
        if progress_bar:
            print('Coregistering..')
        tasks = []
        for t in range(moving.shape[-1]): 
            task_t = dask.delayed(_coreg_t)(
                t, moving, fixed, spacing, p_obj, coreg, transfo, defo
            )
            tasks.append(task_t)
        dask.compute(*tasks)
    else:
        for t in tqdm(
                range(moving.shape[-1]), 
                desc='Coregistering..', 
                disable= not progress_bar,
            ): 
            _coreg_t(
                t, moving, fixed, spacing, p_obj, coreg, transfo, defo
            )
    _cleanup(**params)
    if return_deformation:
        return coreg, transfo, defo
    else:
        return coreg, transfo
     

def _coreg_t(t, moving, fixed, spacing, p_obj, coreg, transfo, defo):
    if defo is None:
        coreg[...,t], transfo[t] = _coreg(
            moving[...,t], fixed[...,t], spacing, p_obj, False
        )
    else:
        coreg[...,t], transfo[t], defo[...,t,:] = _coreg(
            moving[...,t], fixed[...,t], spacing, p_obj, True
        )


def transform_series(
        moving, 
        transfo, 
        path=None, 
        name='deform',
        spacing=1, 
    ):
    """
    Transforms a series of images using a transformation produced by 
    `mdreg.elastix.coreg_series()`.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t).  
    transfo : list
        List of itk.elastixParameterObject with one transform per image 
        in the series.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.
    spacing: array-like
        Pixel spacing in mm. This can be a single scalar if all dimensions 
        are equal, or an array of 2 elements (for 2D data) or 3 elements (
        for 3D data). Defaults to 1.

    Returns
    -------
        numpy.ndarray: The transformed image.
    """
    if not_installed:
        raise ImportError(
            "Coregistration with elastix is optional - please install mdreg as "
            "pip install mdreg[elastix] if you want to use these features."
        )
    
    deform = io._copy(moving, path, name)
    for t, transfo_t in enumerate(transfo):
        deform[...,t] = transform(moving[...,t], transfo_t, spacing=spacing)
    return deform


def coreg(
        moving:np.ndarray, 
        fixed:np.ndarray, 
        spacing=1.0, 
        method='bspline', 
        return_deformation=False,
        **params,
    ):
    """
    Coregister two 2D images or 3D volumes.
    
    Parameters
    ----------
    moving : numpy.ndarray
        The moving image with dimensions (x,y) or (x,y,z). 
    fixed : numpy.ndarray
        The fixed target image with the same shape as the moving image. 
    spacing: array-like
        Pixel spacing in mm. This can be a single scalar if all dimensions 
        are equal, or an array of 2 elements (for 2D data) or 3 elements (
        for 3D data). Defaults to 1. 
    method : str
        Deformation method to use. Options are 'bspline', 'affine', 'rigid' 
        or 'translation'. Default is 'bspline'.
    return_deformation : bool
        If set to True, return the deformation field as a third return value
    params : dict
        Use keyword arguments to overrule any of the default parameters in 
        the elastix template for the chosen method. The default parameters 
        can be found by printing mdreg.elastix.defaults().
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image in the same shape as the moving image.
    transfo : itk.elastixParameterObject
        itk parameter object encoding the transformation from moving to fixed 
        image. The transform can be examined with print(transfo).
    defo : numpy.ndarray
        If requested, the deformation field is returned as a numpy array. 
        The returned displacement vectors are measured in voxel units. 
    """
    if not_installed:
        raise ImportError(
            "Coregistration with elastix is optional - please install mdreg as "
            "pip install mdreg[elastix] if you want to use these features."
        )
    
    if np.sum(np.isnan(moving)) > 0:
        raise ValueError('Moving image contains NaN values - cannot '
                         'perform coregistration')
    if np.sum(np.isnan(fixed)) > 0:
        raise ValueError('Fixed image contains NaN values - cannot '
                         'perform coregistration')

    params_obj = _params_obj(method, **params) 
    return_vals = _coreg(
        moving, fixed, spacing, params_obj, return_deformation,
    )
    _cleanup(**params)
    return return_vals


def _coreg(moving, fixed, spacing, params_obj, return_defo):
    log = False
    # Define spacing
    if np.isscalar(spacing):
        spacing = [spacing] * moving.ndim
    spacing = [np.float64(s) for s in spacing]  
    spacing = list(np.flip(spacing)) 
    if len(spacing) != moving.ndim:
        raise ValueError(f"Spacing must have {moving.ndim} values.") 

    # Create itk images
    moving = np.ascontiguousarray(moving.astype(np.float32))
    fixed = np.ascontiguousarray(fixed.astype(np.float32))
    itk_moving = itk.GetImageFromArray(moving) 
    itk_fixed = itk.GetImageFromArray(fixed) 
    itk_moving.SetSpacing(spacing)
    itk_fixed.SetSpacing(spacing)

    # Perform registration
    try:
        itk_coreg, transfo = itk.elastix_registration_method(
            itk_fixed, itk_moving, parameter_object=params_obj, 
            log_to_console=log,
        )
    except:
        warnings.warn('Elastix coregistration failed. Returning unregistered '
                      'image. To learn more about the error, set log=True.')
        if return_defo:
            return moving.copy(), None, None
        else:
            return moving.copy(), None

    # Return results as numpy arrays
    coreg = itk.GetArrayFromImage(itk_coreg)   

    # Ad-hoc fix for a bug - not sure why this is needed
    if coreg is None:
        coreg = transform(moving, transfo, spacing) 

    if return_defo:
        defo = itk.transformix_deformation_field(itk_moving, transfo)
        defo = itk.GetArrayFromImage(defo)
        defo = np.flip(defo, axis=-1)
        for i, s in enumerate(spacing):
            defo[...,i] = defo[...,i]/s
        return coreg, transfo, defo
    else:
        return coreg, transfo
    

def transform(moving, transfo, spacing=1):
    """
    Transforms an image using a transformation produced by 
    `mdreg.elastix.coreg()`.

    Parameters:
        moving : numpy.ndarray
            The input 2D or 3D image array.
        transfo : itk.elastixParameterObject
            itk parameter object encoding the transformation from moving to fixed 
            image. 
        spacing: array-like
            Pixel spacing in mm. This can be a single scalar if all dimensions 
            are equal, or an array of 2 elements (for 2D data) or 3 elements (
            for 3D data). Defaults to 1. 

    Returns:
        numpy.ndarray: The transformed image.
    """
    if not_installed:
        raise ImportError(
            "Coregistration with elastix is optional - please install mdreg as "
            "pip install mdreg[elastix] if you want to use these features."
        )
    if transfo is None:
        return moving
    
    if np.isscalar(spacing):
        spacing = [spacing] * moving.ndim
    if len(spacing) != moving.ndim:
        raise ValueError(f"spacing needs to have {moving.ndim} elements.") 
    
    # Define spacing
    spacing = [np.float64(s) for s in spacing]  
    spacing = list(np.flip(spacing))
    
    # Create itk image
    moving = np.ascontiguousarray(moving.astype(np.float32))
    moving = itk.GetImageFromArray(moving) 
    moving.SetSpacing(spacing)

    # Perform transformation
    transformed = itk.transformix_filter(moving, transfo)

    # Retur as numpy array
    transformed = itk.GetArrayFromImage(transformed)

    _cleanup()

    return transformed



def _cleanup(
        WriteResultImage = 'true',
        WriteDeformationField = "true", 
        **params):
    
    if WriteDeformationField == 'false':

        try:
            os.remove('deformationField.nii')
        except OSError:
            pass
        try:
            os.remove('deformationField.mhd')
        except OSError:
            pass
        try:
            os.remove('deformationField.raw')
        except OSError:
            pass

        path = os.getcwd()

        try:
            os.remove(os.path.join(path, 'deformationField.nii'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(path, 'deformationField.mhd'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(path, 'deformationField.raw'))
        except OSError:
            pass


def _params_obj(method, **params):
    param_obj = itk.ParameterObject.New() # long runtime ~20s
    param_map = param_obj.GetDefaultParameterMap(method) 
    param_obj.AddParameterMap(param_map)
    for key, val in params.items():
        param_obj.SetParameter(key, str(val))
    return param_obj







# Elastix parameter templates in the form of python dictionaries
# Taken from the defaults in the elastix model zoo
# https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/default


BSPLINE = {
    # Example parameter file for B-spline registration
    # C-style comments: #

    # The internal pixel type, used for internal computations
    # Leave to float in general.
    # NB: this is not the type of the input images! The pixel
    # type of the input images is automatically read from the
    # images themselves.
    # This setting can be changed to "short" to save some memory
    # in case of very large 3D images.
    "FixedInternalImagePixelType": "float",
    "MovingInternalImagePixelType": "float",

    # Specify whether you want to take into account the so-called
    # direction cosines of the images. Recommended: true.
    # In some cases, the direction cosines of the image are corrupt,
    # due to image format conversions for example. In that case, you
    # may want to set this option to "false".
    "UseDirectionCosines": "true",

    # **************** Main Components **************************

    # The following components should usually be left as they are:
    "Registration": "MultiResolutionRegistration",
    "Interpolator": "BSplineInterpolator",
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "Resampler": "DefaultResampler",

    # These may be changed to Fixed/MovingSmoothingImagePyramid.
    # See the manual.
    "FixedImagePyramid": "FixedRecursiveImagePyramid",
    "MovingImagePyramid": "MovingRecursiveImagePyramid",

    # The following components are most important:
    # The optimizer AdaptiveStochasticGradientDescent "ASGD) works
    # quite ok in general. The Transform and Metric are important
    # and need to be chosen careful for each application. See manual.
    "Optimizer": "AdaptiveStochasticGradientDescent",
    "Transform": "BSplineTransform",
    "Metric": "AdvancedMattesMutualInformation",

    # ***************** Transformation **************************

    # The control point spacing of the bspline transformation in
    # the finest resolution level. Can be specified for each
    # dimension differently. Unit: mm.
    # The lower this value, the more flexible the deformation.
    # Low values may improve the accuracy, but may also cause
    # unrealistic deformations. This is a very important setting!
    # We recommend tuning it for every specific application. It is
    # difficult to come up with a good 'default' value.
    "FinalGridSpacingInPhysicalUnits": "16",

    # Alternatively, the grid spacing can be specified in voxel units.
    # To do that, uncomment the following line and comment/remove
    # the FinalGridSpacingInPhysicalUnits definition.
    #(FinalGridSpacingInVoxels 16)

    # By default the grid spacing is halved after every resolution,
    # such that the final grid spacing is obtained in the last
    # resolution level. You can also specify your own schedule,
    # if you uncomment the following line:
    #(GridSpacingSchedule 4.0 4.0 2.0 1.0)
    # This setting can also be supplied per dimension.

    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    "HowToCombineTransforms": "Compose",

    # ******************* Similarity measure *********************

    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #(NumberOfHistogramBins 16 32 64,
    "NumberOfHistogramBins": "32",

    # If you use a mask, this option is important.
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    "ErodeMask": "false",

    # ******************** Multiresolution **********************

    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    "NumberOfResolutions": "4",

    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    # So, in 2D, with 4 resolutions, the following schedule is used:
    #(ImagePyramidSchedule 8 8  4 4  2 2  1 1 )
    # And in 3D:
    #(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
    # You can specify any schedule, for example:
    #"ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
    # Make sure that the number of elements equals the number
    # of resolutions times the image dimension.

    # ******************* Optimizer ****************************

    # Maximum number of iterations in each resolution level:
    # 200-2000 works usually fine for nonrigid registration.
    # The more, the better, but the longer computation time.
    # This is an important parameter!
    "MaximumNumberOfIterations": "500",

    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    #(MaximumStepLength 1.0)

    # **************** Image sampling **********************

    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    "NumberOfSpatialSamples": "2048",

    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    "NewSamplesEveryIteration": "true",
    "ImageSampler": "Random",

    # ************* Interpolation and Resampling ****************

    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most
    # applications a good choice.
    "BSplineInterpolationOrder": "1",

    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    "FinalBSplineInterpolationOrder": "3",

    #Default pixel value for pixels that come from outside the picture:
    "DefaultPixelValue": "0",

    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # not interested in the final deformed moving image, but only
    # want to analyze the deformation field for example.
    "WriteResultImage": "true",

    # The pixel type and format of the resulting deformed moving image
    "ResultImagePixelType": "short",
    "ResultImageFormat": "mhd",

}


AFFINE = {
    # Adapted from:
    # https://github.com/SuperElastix/ElastixModelZoo/blob/master/models/default/Parameters_Affine.txt

    # Example parameter file for affine registration
    # C-style comments: #

    # The internal pixel type, used for internal computations
    # Leave to float in general.
    # NB: this is not the type of the input images! The pixel
    # type of the input images is automatically read from the
    # images themselves.
    # This setting can be changed to "short" to save some memory
    # in case of very large 3D images.
    "FixedInternalImagePixelType": "float",
    "MovingInternalImagePixelType": "float",


    # Specify whether you want to take into account the so-called
    # direction cosines of the images. Recommended: true.
    # In some cases, the direction cosines of the image are corrupt,
    # due to image format conversions for example. In that case, you
    # may want to set this option to "false".
    "UseDirectionCosines": "true",

    # **************** Main Components **************************

    # The following components should usually be left as they are:
    "Registration": "MultiResolutionRegistration",
    "Interpolator": "BSplineInterpolator",
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "Resampler": "DefaultResampler",

    # These may be changed to Fixed/MovingSmoothingImagePyramid.
    # See the manual.
    "FixedImagePyramid": "FixedRecursiveImagePyramid",
    "MovingImagePyramid": "MovingRecursiveImagePyramid",

    # The following components are most important:
    # The optimizer AdaptiveStochasticGradientDescent "ASGD) works
    # quite ok in general. The Transform and Metric are important
    # and need to be chosen careful for each application. See manual.
    "Optimizer": "AdaptiveStochasticGradientDescent",
    "Transform": "AffineTransform",
    "Metric": "AdvancedMattesMutualInformation",

    # ***************** Transformation **************************

    # Scales the affine matrix elements compared to the translations, to make
    # sure they are in the same range. In general, it's best to
    # use automatic scales estimation:
    "AutomaticScalesEstimation": "true",

    # Automatically guess an initial translation by aligning the
    # geometric centers of the fixed and moving.
    "AutomaticTransformInitialization": "true",

    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    "HowToCombineTransforms": "Compose",

    # ******************* Similarity measure *********************

    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #"NumberOfHistogramBins": (16 32 64),
    "NumberOfHistogramBins": "32",

    # If you use a mask, this option is important.
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    "ErodeMask": "false",

    # ******************** Multiresolution **********************

    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    "NumberOfResolutions": "4",

    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    # So, in 2D, with 4 resolutions, the following schedule is used:
    #"ImagePyramidSchedule": "8 8  4 4  2 2  1 1" ,
    # And in 3D:
    #"ImagePyramidSchedule": "8 8 8  4 4 4  2 2 2  1 1 1" ,
    # You can specify any schedule, for example:
    #"ImagePyramidSchedule": "4 4  4 3  2 1  1 1" ,
    # Make sure that the number of elements equals the number
    # of resolutions times the image dimension.

    # ******************* Optimizer ****************************

    # Maximum number of iterations in each resolution level:
    # 200-500 works usually fine for affine registration.
    # For more robustness, you may increase this to 1000-2000.
    "MaximumNumberOfIterations": "250",

    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    #"MaximumStepLength": 1.0,

    # **************** Image sampling **********************

    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    "NumberOfSpatialSamples": "2048",

    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    "NewSamplesEveryIteration": "true",
    "ImageSampler": "Random",

    # ************* Interpolation and Resampling ****************

    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most
    # applications a good choice.
    "BSplineInterpolationOrder": "1",

    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    "FinalBSplineInterpolationOrder": "3",

    #Default pixel value for pixels that come from outside the picture:
    "DefaultPixelValue": "0",

    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # only interested in the final (nonrigidly) deformed moving image
    # for example.
    "WriteResultImage": "true",

    # The pixel type and format of the resulting deformed moving image
    "ResultImagePixelType": "short",
    "ResultImageFormat": "mhd",
}

RIGID = {
    # Adapted from
    # https://github.com/SuperElastix/ElastixModelZoo/blob/master/models/default/Parameters_Rigid.txt
    # Example parameter file for rotation registration
    # C-style comments: #

    # The internal pixel type, used for internal computations
    # Leave to float in general.
    # NB: this is not the type of the input images! The pixel
    # type of the input images is automatically read from the
    # images themselves.
    # This setting can be changed to "short" to save some memory
    # in case of very large 3D images.
    "FixedInternalImagePixelType": "float",
    "MovingInternalImagePixelType": "float",

    # Specify whether you want to take into account the so-called
    # direction cosines of the images. Recommended: true.
    # In some cases, the direction cosines of the image are corrupt,
    # due to image format conversions for example. In that case, you
    # may want to set this option to "false".
    "UseDirectionCosines": "true",

    # **************** Main Components **************************

    # The following components should usually be left as they are:
    "Registration": "MultiResolutionRegistration",
    "Interpolator": "BSplineInterpolator",
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "Resampler": "DefaultResampler",

    # These may be changed to Fixed/MovingSmoothingImagePyramid.
    # See the manual.
    "FixedImagePyramid": "FixedRecursiveImagePyramid",
    "MovingImagePyramid": "MovingRecursiveImagePyramid",

    # The following components are most important:
    # The optimizer AdaptiveStochasticGradientDescent "ASGD) works
    # quite ok in general. The Transform and Metric are important
    # and need to be chosen careful for each application. See manual.
    "Optimizer": "AdaptiveStochasticGradientDescent",
    "Transform": "EulerTransform",
    "Metric": "AdvancedMattesMutualInformation",

    # ***************** Transformation **************************

    # Scales the rotations compared to the translations, to make
    # sure they are in the same range. In general, it's best to
    # use automatic scales estimation:
    "AutomaticScalesEstimation": "true",

    # Automatically guess an initial translation by aligning the
    # geometric centers of the fixed and moving.
    "AutomaticTransformInitialization": "true",

    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    "HowToCombineTransforms": "Compose",

    # ******************* Similarity measure *********************

    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #"NumberOfHistogramBins 16 32 64,
    "NumberOfHistogramBins": "32",

    # If you use a mask, this option is important.
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    "ErodeMask": "false",

    # ******************** Multiresolution **********************

    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    "NumberOfResolutions": "4",

    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    # So, in 2D, with 4 resolutions, the following schedule is used:
    #"ImagePyramidSchedule 8 8  4 4  2 2  1 1 ,
    # And in 3D:
    #"ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 ,
    # You can specify any schedule, for example:
    #"ImagePyramidSchedule 4 4  4 3  2 1  1 1 ,
    # Make sure that the number of elements equals the number
    # of resolutions times the image dimension.

    # ******************* Optimizer ****************************

    # Maximum number of iterations in each resolution level:
    # 200-500 works usually fine for rigid registration.
    # For more robustness, you may increase this to 1000-2000.
    "MaximumNumberOfIterations": "250",

    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    #"MaximumStepLength 1.0,

    # **************** Image sampling **********************

    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    "NumberOfSpatialSamples": "2048",

    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    "NewSamplesEveryIteration": "true",
    "ImageSampler": "Random",

    # ************* Interpolation and Resampling ****************

    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most
    # applications a good choice.
    "BSplineInterpolationOrder": "1",

    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    "FinalBSplineInterpolationOrder": "3",

    #Default pixel value for pixels that come from outside the picture:
    "DefaultPixelValue": "0",

    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # only interested in the final (nonrigidly) deformed moving image
    # for example.
    "WriteResultImage": "true",

    # The pixel type and format of the resulting deformed moving image
    "ResultImagePixelType": "short",
    "ResultImageFormat": "mhd",
}

TRANSLATION = {

    # Example parameter file for translational registration
    # C-style comments: #

    # The internal pixel type, used for internal computations
    # Leave to float in general.
    # NB: this is not the type of the input images! The pixel
    # type of the input images is automatically read from the
    # images themselves.
    # This setting can be changed to "short" to save some memory
    # in case of very large 3D images.
    "FixedInternalImagePixelType": "float",
    "MovingInternalImagePixelType": "float",

    # Specify whether you want to take into account the so-called
    # direction cosines of the images. Recommended: true.
    # In some cases, the direction cosines of the image are corrupt,
    # due to image format conversions for example. In that case, you
    # may want to set this option to "false".
    "UseDirectionCosines": "true",

    # **************** Main Components **************************

    # The following components should usually be left as they are:
    "Registration": "MultiResolutionRegistration",
    "Interpolator": "BSplineInterpolator",
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "Resampler": "DefaultResampler",

    # These may be changed to Fixed/MovingSmoothingImagePyramid.
    # See the manual.
    "FixedImagePyramid": "FixedRecursiveImagePyramid",
    "MovingImagePyramid": "MovingRecursiveImagePyramid",

    # The following components are most important:
    # The optimizer AdaptiveStochasticGradientDescent (ASGD) works
    # quite ok in general. The Transform and Metric are important
    # and need to be chosen careful for each application. See manual.
    "Optimizer": "AdaptiveStochasticGradientDescent",
    "Transform": "TranslationTransform",
    "Metric": "AdvancedMattesMutualInformation",

    # ***************** Transformation **************************

    # The following option does not apply to the Translation transform,
    # but it will be simply ignored.
    # See Rigid and Affine parameter files for an explanation.
    "AutomaticScalesEstimation": "true",

    # Automatically guess an initial translation by aligning the
    # geometric centers of the fixed and moving.
    "AutomaticTransformInitialization": "true",

    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    "HowToCombineTransforms": "Compose",

    # ******************* Similarity measure *********************

    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #"NumberOfHistogramBins 16 32 64,
    "NumberOfHistogramBins": "32",

    # If you use a mask, this option is important.
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    "ErodeMask": "false",

    # ******************** Multiresolution **********************

    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    "NumberOfResolutions": "4",

    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    # So, in 2D, with 4 resolutions, the following schedule is used:
    #"ImagePyramidSchedule 8 8  4 4  2 2  1 1 ,
    # And in 3D:
    #"ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 ,
    # You can specify any schedule, for example:
    #"ImagePyramidSchedule 4 4  4 3  2 1  1 1 ,
    # Make sure that the number of elements equals the number
    # of resolutions times the image dimension.

    # ******************* Optimizer ****************************

    # Maximum number of iterations in each resolution level:
    # 200-500 works usually fine for rigid registration.
    # For more robustness, you may increase this to 1000-2000.
    "MaximumNumberOfIterations": "250",

    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    #"MaximumStepLength 1.0,

    # **************** Image sampling **********************

    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    "NumberOfSpatialSamples": "2048",

    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    "NewSamplesEveryIteration": "true",
    "ImageSampler": "Random",

    # ************* Interpolation and Resampling ****************

    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most
    # applications a good choice.
    "BSplineInterpolationOrder": "1",

    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    "FinalBSplineInterpolationOrder": "3",

    #Default pixel value for pixels that come from outside the picture:
    "DefaultPixelValue": "0",

    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # only interested in the final (nonrigidly) deformed moving image
    # for example.
    "WriteResultImage": "true",

    # The pixel type and format of the resulting deformed moving image
    "ResultImagePixelType": "short",
    "ResultImageFormat": "mhd",
}






# def _coreg_3d(source_large, target_large, spacing, downsample, log, mask, 
#               params_obj):
    
#     if np.sum(np.isnan(source_large)) > 0:
#         raise ValueError('Moving image contains NaN values - cannot '
#                          'perform coregistration')
#     if np.sum(np.isnan(target_large)) > 0:
#         raise ValueError('Target image contains NaN values - cannot '
#                          'perform coregistration')
#     if np.isscalar(spacing):
#         spacing = [spacing, spacing, spacing]
#     if len(spacing) != 3:
#         raise ValueError("For 3D registration, spacing must be a 3D array.") 
    
#     # Define spacing and origin
#     spacing = [np.float64(s) for s in spacing]  
#     spacing_large = [spacing[2], 
#                      spacing[1], 
#                      spacing[0]] # correct numpy(x,y,z) ordering for itk(z,y,x)
#     spacing_large_z, spacing_large_y, spacing_large_x = spacing_large 
#     origin_large = [0,0,0]

#     # Downsample source and target
#     target_small = block_reduce(target_large, block_size=downsample, func=np.mean)
#     source_small = block_reduce(source_large, block_size=downsample, func=np.mean)
#     spacing_small = [spacing_large[0]*downsample, 
#                      spacing_large[1]*downsample, 
#                      spacing_large[2]*downsample] # downsample large spacing
#     spacing_small_z, spacing_small_y, spacing_small_x = spacing_small
#     origin_small = [(spacing_small_z - spacing_large_z) / 2, 
#                     (spacing_small_y - spacing_large_y) / 2, 
#                     (spacing_small_x - spacing_large_x) / 2]

#     # Coregister downsampled source to target
#     source_small = np.ascontiguousarray(source_small.astype(np.float32))
#     target_small = np.ascontiguousarray(target_small.astype(np.float32))
#     source_small = itk.GetImageFromArray(source_small) 
#     target_small = itk.GetImageFromArray(target_small) 
#     source_small.SetSpacing(spacing_small)
#     target_small.SetSpacing(spacing_small)
#     source_small.SetOrigin(origin_small)
#     target_small.SetOrigin(origin_small)
#     try:
#         coreg_small, result_transform_parameters = itk.elastix_registration_method(
#             target_small, source_small,
#             parameter_object=params_obj, 
#             log_to_console=log) # perform registration of downsampled image
#     except:
#         warnings.warn('Elastix coregistration failed. Returning zero '
#                       'deformation field. To find out the error, set log=True.')
#         deformation_field = np.zeros(source_large.shape + (len(source_large.shape), ))
#         return source_large.copy(), deformation_field
    
#     # Get coregistered image at original size
#     large_shape_x, large_shape_y, large_shape_z = source_large.shape
#     size = [str(large_shape_z), str(large_shape_y), str(large_shape_x)]
#     space = [str(spacing_large_z), str(spacing_large_y), str(spacing_large_x)]
#     result_transform_parameters.SetParameter(0, "Size", size)
#     result_transform_parameters.SetParameter(0, "Spacing", space)
#     source_large = np.ascontiguousarray(source_large.astype(np.float32))
#     source_large = itk.GetImageFromArray(source_large)
#     source_large.SetSpacing(spacing_large)
#     source_large.SetOrigin(origin_large)
#     coreg_large = itk.transformix_filter(
#         source_large,
#         result_transform_parameters,
#         log_to_console=log)


#     return coreg_large, result_transform_parameters




# def _coreg_2d(source_large, target_large, spacing, downsample, log, mask, 
#               params_obj):
    
#     if np.sum(np.isnan(source_large)) > 0:
#         raise ValueError('Moving image contains NaN values - cannot '
#                          'perform coregistration')
#     if np.sum(np.isnan(target_large)) > 0:
#         raise ValueError('Target image contains NaN values - cannot '
#                          'perform coregistration')

#     if np.isscalar(spacing):
#         spacing = [spacing, spacing]
#     if len(spacing) != 2:
#         raise ValueError("For 2D registration, spacing must be a 2D array.")
    
#     # Define spacing and origin
#     spacing = [np.float64(s) for s in spacing]
#     spacing_large = [spacing[1], spacing[0]] # correct numpy(x,y) ordering for itk(y,x)
#     spacing_large_y, spacing_large_x = spacing_large 
#     origin_large = [0,0]

#     # Downsample source and target
#     target_small = block_reduce(target_large, block_size=downsample, func=np.mean)
#     source_small = block_reduce(source_large, block_size=downsample, func=np.mean)
#     spacing_small = [spacing_large[0]*downsample, 
#                         spacing_large[1]*downsample] # downsample large spacing
#     spacing_small_y, spacing_small_x = spacing_small
#     origin_small = [(spacing_small_y - spacing_large_y) / 2, 
#                     (spacing_small_x - spacing_large_x) / 2]

#     # Coregister downsampled source to target
#     source_small = np.ascontiguousarray(source_small.astype(np.float32))
#     target_small = np.ascontiguousarray(target_small.astype(np.float32))
#     source_small = itk.GetImageFromArray(source_small) 
#     target_small = itk.GetImageFromArray(target_small)
#     source_small.SetSpacing(spacing_small)
#     target_small.SetSpacing(spacing_small)
#     source_small.SetOrigin(origin_small)
#     target_small.SetOrigin(origin_small)
#     try:
#         coreg_small, result_transform_parameters = itk.elastix_registration_method(
#             target_small, source_small,
#             parameter_object=params_obj, 
#             log_to_console=log)
#     except:
#         warnings.warn('Elastix coregistration failed. Returning zero '
#                       'deformation field. To find out the error, set log=True.')
#         deformation_field = np.zeros(source_large.shape + (len(source_large.shape), ))
#         return source_large.copy(), deformation_field
    
#     # Get coregistered image at original size
#     large_shape_x, large_shape_y = source_large.shape
#     size = [str(large_shape_y), str(large_shape_x)]
#     space = [str(spacing_large_y), str(spacing_large_x)]
#     result_transform_parameters.SetParameter(0, "Size", size)
#     result_transform_parameters.SetParameter(0, "Spacing", space)
#     source_large = np.ascontiguousarray(source_large.astype(np.float32))
#     source_large = itk.GetImageFromArray(source_large)
#     source_large.SetSpacing(spacing_large)
#     source_large.SetOrigin(origin_large)
#     coreg_large = itk.transformix_filter(
#         source_large,
#         result_transform_parameters,
#         log_to_console=log)
#     coreg_large = itk.GetArrayFromImage(coreg_large)

#     return coreg_large, result_transform_parameters



