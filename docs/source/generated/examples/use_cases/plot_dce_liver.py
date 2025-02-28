"""
===========================================================
3D DCE-MRI in the liver
===========================================================

Fit a linear 2-compartment model to a 3D DCE dataset.
"""

#%% 
# Setup
# -----

import os
import time
import shutil
import numpy as np
import mdreg

# We use zarray data in this example
data = mdreg.fetch_zarr('DCE_small')

# Variables used in this script
tacq = data.attrs['time'] 
aif = data.attrs['aif']  
spacing = data.attrs['spacing']

# Path for output
results_path = os.path.join(os.getcwd(), 'tmp')

# Check the data
anim = mdreg.plot.animation(
    #data[:,:,20:24,100:150], 
    data, 
    title='DCE Data', 
    vmin=0,
    vmax=0.9*np.max(data[...,0]),
)

#%%
# Perform motion correction
# -------------------------
# We fit the DCE data using a linearised 2-compartment model, using motion 
# correction with default settings:

t = time.time()

coreg, fit, transfo, pars = mdreg.fit(
    data,
    fit_image={
        'func': mdreg.fit_2cm_lin,                
        'time': tacq,                   # Acquisition times 
        'aif': aif,                     # Signal-time curve in the aorta
        'baseline': 5,                  # Nr of precontrast samples 
    },
    fit_coreg = {
        'package': 'elastix',
        'spacing': spacing,
        'FinalGridSpacingInPhysicalUnits': 50.0,
    },
    maxit = 3,
    path = results_path,
    verbose=2,
)

print(f"Computation time: {round(time.time()-t)} seconds.")

#%% 
# Check the result
anim = mdreg.plot.animation(
    #coreg[:,:,20:24,100:150],
    coreg, 
    title='DCE motion corrected', 
    vmin=0,
    vmax=0.9*np.max(data[...,0]),
)

#%%
# Cleanup disk
shutil.rmtree(results_path)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -2

# sphinx_gallery_end_ignore