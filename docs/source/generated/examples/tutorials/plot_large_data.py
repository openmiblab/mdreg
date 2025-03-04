"""
===========================================================
Datasets larger than RAM
===========================================================

When data are entered as numpy arrays, mreg will load the entire array in memory, 
perform the computations and generate new datasets in memory, such as model fit 
arrays or deformation fields. 

When working with large datasets, such as 3D time series of images, it may 
happen that this requires more memory than a machine has available. Even 
if the original data fit in memory, you will need more than three times that 
to fit all the results in. 

A solution in such conditions is to store the data on disk as a zarray rather 
than a numpy array. Most mdreg functions can operate on zarrays, and will 
consume a lot less memory because only the data that are actually being 
processed are loaded. 

This example illustrates how to work with zarrays for the example of 
non-linear model fitting to a variable flip angle series.

"""

#%% 
# Setup
# -----

import os
import time
import shutil
import numpy as np
import mdreg

#%%
# Fetch the low resolution VFA data saved as zarray. Since the zarray 
# exist on disk this step is not actually loading any data in memory.
data = mdreg.fetch_zarr('VFA')

#%%
# In a zarray any header information or metadata are in attributes:
FA = data.attrs['FA']             # The FA values in degrees

#%%
# For the purpose of this example we will always use the same pixel-wise 
# model fit. We define the keyword arguments up front to save some repetition later:
modelfit = {                           
    'model': mdreg.spgr_vfa,            # VFA signal model
    'xdata': FA,                        # Flip angle 
    'func_init': mdreg.spgr_vfa_init,   # Initializer
    'p0': [1, 0.5],                     # Initial values
    'bounds': ([0, 0], [np.inf, 1]),    # Parameter bounds
}

#%%
# Fit numpy arrays
# ----------------
# Loading an entire array into memory can be done by indexing with data[:], 
# which returns a numpy array. Therefore the following operation simply fits 
# the signal model to a numpy array in memory:
t = time.time()
coreg, fit, defo, pars = mdreg.fit(
    data[:], 
    fit_pixels=modelfit, 
    maxit=1,
)
print(f"Computation time: {round(time.time()-t)} seconds.")

#%%
# Since the argument is a numpy array, the return values are numpy arrays too:
print('Data type of return values: ', type(fit))

#%% 
# Let's visualise the S0 map for reference:

#%%
fig = mdreg.plot.par(pars[...,0], title='S0', vmin=0, 
                     vmax=np.percentile(pars[...,0], 95))

#%%
# Fitting zarrays: results in memory
# ----------------------------------
# We can also feed in the zarray directly as argument:
t = time.time()
coreg, fit, defo, pars = mdreg.fit(
    data, 
    fit_pixels=modelfit, 
    maxit=1,
)
print(f"Computation time: {round(time.time()-t)} seconds.")

#%%
# This effectively performs the same operation on the same data in memory. 
# The main difference is that the return values are now also zarrays:
print('Data type of return values: ', type(fit))

#%% 
# The result is the same as with numpy arrays, and can be accessed in the 
# same way:

#%%
fig = mdreg.plot.par(pars[...,0], title='S0', vmin=0, 
                     vmax=np.percentile(pars[...,0], 95))

#%%
# Fitting zarrays: results on disk
# --------------------------------
# Since we did not specify a path before, the zarrays returned by mfit() are 
# stored in memory:
print('Storage type of return values: ', type(fit.store))

#%%
# If we want the return values to be stored on disk instead, we need to 
# provide a path in the function call:
path = os.path.join(os.getcwd(), 'tmp')

t = time.time()
coreg, fit, defo, pars = mdreg.fit(
    data, 
    path=path, 
    fit_pixels=modelfit, 
    maxit=1,
)
print(f"Computation time: {round(time.time()-t)} seconds.")

#%%
# This has now created four zarrays on disk:
print('Storage type of return values: ', type(fit.store))

#%%
# Clean up the directory for the next computation
shutil.rmtree(path)

#%%
# Row-by-row computation
# ----------------------
# By default zarrays are processed slice-by-slice (memdim=2). 
# If a single slice is still too large for memory, the *memdim* argument can
# be set to 1 to perform the computations row-by-row:
modelfit['memdim']=1

t = time.time()
coreg, fit, defo, pars = mdreg.fit(
    data, 
    path=path, 
    fit_pixels=modelfit,
    maxit=1,
)
print(f"Computation time: {round(time.time()-t)} seconds.")

#%%
# Clean up directory
shutil.rmtree(path)

#%%
# Pixel-by-pixel computation
# --------------------------
# Computations can also be performed pixel-by-pixel by setting *memdim* to 0, 
# but this will come at a major cost in computation time as a complete slice 
# needs to be read, and written, for each pixel. 


# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore