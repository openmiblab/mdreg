"""
===========================================================
Fitting models without motion correction
===========================================================

This example illustrates how mdreg can be used to fit models without 
motion correction. The idea is illustrated for a 3D time series with 
variable flip angles (VFA), and a linear signal model fit.
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

#%%
# Perform model fit
# -----------------
# The linear VFA signal model is included in `mdreg` as the function 
# `mdreg.fit_spgr_vfa_lin`, which requires the flip angle (FA) values in 
# degrees as input:

fit, pars = mdreg.fit_spgr_vfa_lin(
    array,                      # Signal data to fit
    FA=FA,                      # Flip angle in degrees  
    progress_bar=True,          # Set to True to show a progress bar
)

#%% 
# Plot the model parameters:
S0 = pars[...,0]
fig = mdreg.plot.par(S0, title='S0', vmin=0, vmax=5*np.amax(array))

#%%
T1 = -1/np.log(pars[...,1])
fig = mdreg.plot.par(T1, title='T1/TR', vmin=0, vmax=500)

#%% 
# Check the model fit:
anim = mdreg.plot.animation(
    fit, 
    title='VFA model fit', 
    vmin=0,
    vmax=np.percentile(array, 99),
)

#%%
# Pixel-by-pixel fitting
# ----------------------
# Alternatively, the function `mdreg.fit_pixels` can be used to fit any 
# single-pixel model directly:

fit, pars = mdreg.fit_pixels(
    array,
    model=mdreg.spgr_vfa,
    xdata=FA,
    func_init=mdreg.spgr_vfa_init,
    p0=[1,0.5],
    bounds=([0,0], [np.inf,1]),
)

#%% 
# Check the model fit:
anim = mdreg.plot.animation(
    fit, 
    title='VFA model fit', 
    vmin=0,
    vmax=np.percentile(array, 99),
)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -2

# sphinx_gallery_end_ignore