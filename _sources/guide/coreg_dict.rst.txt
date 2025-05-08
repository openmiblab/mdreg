.. _coreg_dict:

*****************************
Coregistration options
*****************************

This section provides information about the arguments that can be
passed to `mdreg.fit` to control the coregistration process. These arguments 
are provided as a dictionary referred to as `fit_coreg` in the `mdreg.fit` 
function.

skimage
-------

If the `fit_coreg` dictionary is not provided in the call to `mdreg.fit`, 
then `mdreg` uses the `skimage` package for coregistration as default - 
specifically the function `skimage.registration.optical_flow_tvl1` with 
default parameters. 

This behaviour can be modified by providing additional items to the 
`fit_coreg` dictionary. All keyword arguments accepted by the function 
`mdreg.skimage.coreg_series` can be provided. 

For instance, the following call to `mdreg.fit` will show a progress bar 
during coregistration and applies the registration with the `attachment` 
parameter set to a value of 30 instead of the default 15:

.. code-block:: python

    coreg, _, _, _ = mdreg.fit(
        array, 
        fit_coreg = {
          'progress_bar': True,
          'attachment': 30,
        },
    )


elastix
-------

Elastix is an alternative coregistration engine to perform 
the coregistration components in `mdreg`. For consistent usage  
across different coregistration engines, `mdreg` contains pythonic wrappers 
for the core functions in elastix, but the functionality is 
not otherwise modified. 

To run coregistration with `elastix`, the package needs to be specified along 
with any keyword arguments accepted by `mdreg.elastix.coreg_series`. For 
instance the following call will run elastix with a grid spacing of 50mm 
between the control points:

.. code-block:: python

    coreg, _, _, _ = mdreg.fit(
        array, 
        fit_coreg = {
          'package': 'elastix',
          'spacing': [1.25, 1.25, 3.0],
          'FinalGridSpacingInPhysicalUnits': 50.0,
        },
    )

We refer to the original 
`elastix pages <https://github.com/SuperElastix>`_ 
for more detail on elastix. Please note elastix authors request that the 
following papers are cited if you use the elastix software anywhere:

- S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, 
  "elastix: a toolbox for intensity based medical image registration,
  " IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, 
  January 2010. 

- D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, 
  "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification 
  of Alzheimer’s Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, 
  pp. 1-15, January 2014. 

`mdreg` uses the interface `itk-elastix` which is based on SimpleElastix, 
created by Kasper Marstal:

- Kasper Marstal, Floris Berendsen, Marius Staring and Stefan Klein, 
  "SimpleElastix: A user-friendly, multi-lingual library for medical image 
  registration", International Workshop on Biomedical Image Registration 
  (WBIR), Las Vegas, Nevada, USA, 2016.


antspyx
-------

The final registration package wrapped in `mdreg` is *ANTs* through the 
python package `antspyx`. As for elastix, in order to use *ANTs* for 
coregistration it suffices to provide the package name and any keyword 
arguments accepted by `mdreg.ants.coreg_series`:

.. code-block:: python

    coreg, _, _, _ = mdreg.fit(
        array, 
        fit_coreg={
            'package': 'ants',
            'type_of_transform': 'SyNOnly',
        },
    )

This will call the function 
`ants.registration <https://antspy.readthedocs.io/en/latest/registration.html>`_ 
to coregister the dynamics.
