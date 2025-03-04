"""
===============================================
Multi-slice 2D registration
===============================================

By default mdreg performs 3D coregistration on 3D series, and 2D 
coregistration on 2D series. In case the 3D series encodes a collection of 
2D slices with gaps in between (multi-slice data), this may not be appropriate.

This examples shows how mdreg can be used to perform 2D coregistration 
slice-by-slice on a 3D series by setting the keyword argument force_2d=True.
"""

#%%
# Setup
# -----
import numpy as np
import mdreg

#%%
# Fetch the multi-slice MOLLI dataset
data = mdreg.fetch('MOLLI')

# Get the relevant variables (3D data)
array = data['array'] 
TI = np.array(data['TI'])/1000 


#%%
# Perform slice-by-slice motion correction
# ----------------------------------------

# Define the signal model
molli = {
    'func': mdreg.fit_abs_exp_recovery_2p, 
    'TI': np.array(data['TI'])/1000,
}
# Perform motion correction
coreg, fit, defo, pars = mdreg.fit(
    array, 
    fit_image=molli, 
    force_2d=True, 
    maxit=1,
)

# %%
# Since coregistration is performed in 2D, the deformation field only has 
# two components:
print(f'The deformation field has {defo.shape[-1]} components.')

#%%
# Visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4)

#%%
# Different options per slice
# ---------------------------
# It is not uncommon that each slice in a multislice sequence has different 
# imaging parameters. For instance in a MOLLI sequence such as used 
# in this example, it is often the case that each slice has its own set of TI
# values. 
# 
# The situation can be accomodated in `~mdreg.fit` by assigning a 
# list of dictionaries to the *fit_image* argument - one for each slice. 
# We illustrate that here assuming that the TI's for each slice are offset by 
# 100 ms relative to the previous slice:

molli = [
    {
        'func': mdreg.fit_abs_exp_recovery_2p, 
        'TI': TI + 0.1 * z, 
    }
    for z in range(array.shape[2])
]

# Other than that, the function call to `mdreg.fit()` is the same as before.

#%%
# Other coregistration packages
# -----------------------------
# This works the same when using ants or elastix for coregistration:

# Perform motion correction
coreg, fit, transfo, pars = mdreg.fit(
    array, 
    fit_image=molli, 
    fit_coreg={'package': 'ants'},
    force_2d=True, 
    maxit=1,
)

#%%
# The difference with skimage is that the transformations returned are now a 
# 2D array with one transformation at each time and each slice:
print(f'2D transformation shape: {transfo.shape}')

# %%
# Compare this to 3D registration, where one 3D transformation for each
# time point is returned:

coreg, fit, transfo, pars = mdreg.fit(
    array, 
    fit_image=molli[0], 
    fit_coreg={'package': 'elastix'},
    maxit=1,
)

print(f'3D transformation shape: {transfo.shape}')


#%%
# Slice by slice with pixel models
# --------------------------------
# If the signal model fit is defined with a custom model via the 
# *fit_pixels* argument, the slice-by-slice operation works the same: 
# set the *force_2d* keyword to True, and - if each slice has 
# different parameter settings - supply the *fit_pixels* 
# argument as a list of dictionaries.

molli = [
    {
        'model': mdreg.abs_exp_recovery_2p, 
        'xdata': TI + 0.1 * z, 
        'func_init': mdreg.abs_exp_recovery_2p_init,
        'p0': [1, 1],
    }
    for z in range(array.shape[2])
]

coreg, fit, transfo, pars = mdreg.fit(
    array, 
    fit_pixels=molli, 
    force_2d=True,
    maxit=1,
)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore