"""
===============================================
3D Variable Flip Angle (Linear)
===============================================

This example illustrates motion correction of a 3D time series with 
variable flip angles (VFA). The motion correction is performed with 3D 
coregistration and using a linear signal model fit.

"""

#%% 
# Setup
# -----
import numpy as np
import mdreg

# Example data included in mdreg
data = mdreg.fetch('VFA')

# Variables used in this examples
array = data['array']       # 4D signal data (x, y, z, FA)
FA = data['FA']             # The FA values in degrees
spacing = data['spacing']   # (x,y,z) voxel size in mm.


#%%
# Perform motion correction
# -------------------------
# The signal model above is included in `mdreg` as the function 
# `mdreg.fit_spgr_vfa_lin`, which require the flip angle (FA) values in degrees as 
# input:

fit_params = {
    'func': mdreg.fit_spgr_vfa_lin,     # VFA signal model
    'FA': FA,                           # Flip angle in degress  
}

#%%
# For coregistration we will use elastix and a relatively coarse deformation 
# field with grid spacing 50mm. We also ask to return the deformation field 
# so we can inspect it:

coreg_params = {
    'package': 'elastix',
    'spacing': spacing,
    'FinalGridSpacingInPhysicalUnits': 50.0,
    'return_deformation': True,
}

#%% 
# We can now perform the motion correction:

coreg, fit, transfo, pars, defo = mdreg.fit(
    array,                          # Signal data to correct
    fit_image = fit_params,         # Signal model
    fit_coreg = coreg_params,       # Coregistration model
    maxit = 2,                      # Maximum number of iteration
    verbose = 2,
)

#%% 
# Visualize the results
# ---------------------
# We visualise the original data and results of the computation using the 
# builtin `animation` function. Since we want to call this 3 times, 
# we define the settings up front:

plot_settings = {
    'vmin' : 0,                         # Minimum value of the colorbar
    'vmax' : np.percentile(array,99),   # Maximum value of the colorbar
}

#%% 
# Now we can plot the data, coregistered images and model fits separately:

#%%
anim = mdreg.plot.animation(array, title='Original data', **plot_settings)

#%%
anim = mdreg.plot.animation(coreg, title='Motion corrected', **plot_settings)

#%%
anim = mdreg.plot.animation(fit, title='Model fit', **plot_settings)

#%% 
# It's also instructive to show the deformation field and check whether 
# deformations are consistent with the effect of breathing motion. Since the 
# deformation field is a vector we show here its norm:

#%%

# Get the norm of the deformation field and adjust the plot settings
defo = mdreg.defo_norm(defo)
plot_settings['vmax'] = np.percentile(defo, 90)

# Display the norm of the deformation field
anim = mdreg.plot.animation(defo, title='Deformation field', **plot_settings)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore