"""
===============================================
Coregistration options in mdreg
===============================================

By default, ``mdreg`` performs coregistration using the optical flow method 
:func:`~skimage.optical_flow_tvl1` as implemented in skimage, with default 
settings for all parameters. 

The coregistration can be modified either 
by overriding these defaults, or by coregistering with one of the optional 
coregistration engines wrapped in mdreg: *ants* and *elastix*.

This example shows how these options can be customized to achieve the best 
results.
"""

#%%
# Setup
# -----
import numpy as np
import mdreg

#%%
# Load test data
data = mdreg.fetch('MOLLI')
array = data['array'][:,:,0,:]

# Throughout this example we use the same signal model:
molli = {
    'func': mdreg.fit_abs_exp_recovery_2p,
    'TI': np.array(data['TI'])/1000,
}

# Visualise the data
anim = mdreg.plot.animation(array, vmin=0, vmax=1e4)


# %%
# Coregistration with ``skimage``
# -------------------------------
# If we don't provide any detail on the coregistration method, ``mdreg`` 
# uses the optical flow method :func:`~skimage.optical_flow_tvl1` from the 
# package ``skimage`` for coregistration:

coreg, fit, _, _ = mdreg.fit(
    array, 
    fit_image=molli,
)

#%%
# Visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4)

# %%
# This has clearly had some effect, but the motion correction is not perfect 
# and the coregistered image is somewhat blurred. We could try to improve 
# the result by optimizing the parameters. 
# 
# One parameter in 
# :func:`~skimage.optical_flow_tvl1` is the *attachment*, which by default is 
# set to 15, but can be set to smaller values for smoother results. In this 
# case we want the opposite, so let's see what happens when we set it to 
# 30. 
# 
# We can do this by providing the *fit_coreg* argument to *mdreg.fit*:

coreg, fit, _, _ = mdreg.fit(
    array, 
    fit_image=molli,
    fit_coreg={'attachment': 30},
)

# Visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4)

#%%
# This appears to have had the desired effect and has resulted in a better 
# motion correction.


#%%
# Coregistration with elastix
# ---------------------------
# An other approach to optimization is to use another coregistration engine. 
# ``mdreg`` also wraps elastix, so let's try that.
#
# By default `mdreg` applies ``elastix`` with bspline deformations, and default settings for all 
# parameters. The result can be further customized by providing any valid 
# ``elastix`` parameter in the dictionary. For this example we will change the 
# default grid spacing from 16mm to 50mm. Since elastix is not default, we 
# need to provide the package name as well. It also needs the pixel spacing 
# so that dimensions in mm are well defined:

coreg, fit, _, _ = mdreg.fit(
    array, 
    fit_image=molli,
    fit_coreg={
        'package': 'elastix',
        'spacing': data['pixel_spacing'],
        'FinalGridSpacingInPhysicalUnits': 50.0,
    },
)

# Visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4)


#%%
# Coregistration with ants
# ---------------------------
# Apart from ``skimage`` and ``elastix``, mdreg also wraps ``ants`` as a 
# registration engine.
#
# It can be called like elastix, by specifying the package and, if needed, 
# any of the parameters to be set to different values than the default. We 
# illustrate this here by coregistering with a transform type different 
# from the default:

coreg, fit, _, _ = mdreg.fit(
    array, 
    fit_image=molli,
    fit_coreg={
        'package': 'ants',
        'type_of_transform': 'SyNOnly',
    },
)

#%%
# Looking at the results, this is not a great solution for these data.
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4)


# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore