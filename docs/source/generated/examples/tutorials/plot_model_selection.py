"""
===============================================
The effect of model selection
===============================================

Since ``mdreg`` performs model-driven motion correction, the choice of an 
appropriate model is important to the result. We illustrate this here by
coregistering a dataset with different models.
"""

#%%
# Setup
# -----
import numpy as np
import mdreg

#%%
# fetch test data

data = mdreg.fetch('MOLLI')

# We will consider the slice z=0 of the data array:
array = data['array'][:,:,0,:]

#%%
# Default model
# -------------
# The breathing motion is clearly visible in this slice and we can use 
# ``mdreg`` to remove it. As a starting point, we could try ``mdreg`` with 
# default settings.

# Perform model-driven coregistration with default settings
coreg, fit, defo, pars = mdreg.fit(array)

# And visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 


# %%
# The default model is a constant, so the model fit (left) does not show any 
# changes. The coregistered image has not properly removed the motions. This 
# is not unexpected, because a constant model does not provide a good 
# approximation to the changes in image contrast. We clearly need a 
# more complex model for this sequence.

#%%
# Linear model
# ------------
# In order to improve on this result, we could try a linear model, which 
# approximates the signal changes in each pixel as a straight line. 

# Perform model-driven coregistration with default settings
coreg, fit, defo, pars = mdreg.fit(
    array,
    fit_pixels = {
        'model': mdreg.lin,
        'p0': [1, 0],
    },
)
# And visualise the results
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 

#%%
# Still not a great motion correction (right). The model fit (left) shows that 
# while the linear model does allow for some changes in contrast over time, 
# it does not capture the actual changes very well (middle). 

#%%
# Quadratic model
# ---------------
# Let's step up the complexity once again and fit with a quadratic model: 
coreg, fit, defo, pars = mdreg.fit(
    array,
    fit_pixels = {
        'model': mdreg.quad,
        'p0': [1, 0, 0],
    },
)
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 

#%%
# This now captures the signal changes better, leading to an improved motion 
# correction, but the result is far from perfect. 

#%%
# Fourth order polynomial
# -----------------------
# Let's step it up one more time to see if we can improve on this further. 
# We'll skip a step and go straight to fourth order:
coreg, fit, defo, pars = mdreg.fit(
    array,
    fit_pixels = {
        'model': mdreg.ofour,
        'p0': [1, 0, 0, 0, 0],
    },
)
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 

# %%
# This now appears to have made it worse: there is more motion again in the 
# coregistered series (right). Looking at the model fit (left) we see what is 
# happening: this model has so much freedom that it can now model the 
# deformations as well, creating a moving target for the coregistration. 
#
# The best solution, when available, is always to use the actual model of the 
# signal changes, with the smallest amount of free parameters as is needed to 
# describe them accurately.

# %%
# MOLLI model
# -----------
# We will run this one final time, now using the correct model for a Look-
# Locker MRI signal sequence. Tis only has 2 parameters, but models the signal 
# changes well:
coreg, fit, defo, pars = mdreg.fit(
    array,
    fit_pixels = {
        'model': mdreg.abs_exp_recovery_2p,
        'p0': [1, 1],
        'xdata': np.array(data['TI'])/1000,
        'func_init':mdreg.abs_exp_recovery_2p_init,
    },
)
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 

#%%
# This show the best result so far, despite the model only having 2 free 
# parameters. At this point the result cannot be improved by fine tuning 
# the modelling, but changing the restrictions in the default coregistration 
# does help to improve further: 
coreg, fit, defo, pars = mdreg.fit(
    array,
    fit_pixels = {
        'model': mdreg.abs_exp_recovery_2p,
        'p0': [1, 1],
        'xdata': np.array(data['TI'])/1000,
        'func_init':mdreg.abs_exp_recovery_2p_init,
    },
    fit_coreg = {
        'attachment': 30,
    },
)
anim = mdreg.plot.series(array, fit, coreg, vmin=0, vmax=1e4) 

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore