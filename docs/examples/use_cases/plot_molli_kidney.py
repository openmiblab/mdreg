"""
===============================================
Multi-slice MOLLI T1-mapping in the kidney
===============================================

This example shows how mdreg can be used to perform 2D motion correction 
slice-by-slice on a 4D array, for the case of Look-Locker T1 mapping.
"""

#%%
# Setup
# -----
import numpy as np
import mdreg

# Fetch the multi-slice MOLLI dataset and plot
data = mdreg.fetch('MOLLI')

# Get the relevant variables
array = data['array']  

# Visualise the motion
anim = mdreg.plot.animation(array, vmin=0, vmax=1e4)

#%%
# Perform slice-by-slice motion correction
# ----------------------------------------
# Slice-by-slice analysis works the same way as single-slice or 3D analysis. We 
# just have to remember to set the keyword argument *force_2d* to True so 
# `~mdreg.fit` knows that we want 2D motion correction. This overrules the 
# default behaviour of fitting 3D data with 3D motion correction:

coreg, fit, transfo, pars = mdreg.fit(
    array, 
    fit_image={
        'func': mdreg.fit_abs_exp_recovery_2p, 
        'TI': np.array(data['TI'])/1000,
    }, 
    fit_coreg={
        'package': 'elastix',
        'spacing': data['pixel_spacing'],
        'FinalGridSpacingInPhysicalUnits': 50.0,
    },
    force_2d=True, 
    verbose=2,
)

#%%
# Visualise the results

anim = mdreg.plot.animation(
    coreg, title='Motion corrected', interval=500, vmin=0, vmax=1e4
)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore