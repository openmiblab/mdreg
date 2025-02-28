"""
==============================================
Pairwise coregistration API
==============================================

``mdreg`` includes a harmonized API for pairwise coregistration of 2D images
or 3D volumes with three different packages: ``ants``, ``skimage`` and 
``itk-elastix``. 

This examples illustrates their usage for the example of pairwise 3D 
registration.
"""

#%%
# Setup
# -----
import os
import time
import numpy as np
import mdreg


#%%
# Load test data
data = mdreg.fetch('VFA')

# Select first and last volume to coregister
fixed = data['array'][:,:,:,0]
moving = data['array'][:,:,:,-1]

# Relevant header data
FA = data['FA']             # The FA values in degrees
spacing = data['spacing']   # (x,y,z) voxel size in mm.

#%%
# Check alignment
# ---------------
# On the difference image, the effect of breathing motion can be clearly seen 
# as a white line at the edge of the liver:

# Difference image
diff = fixed - moving

# Keep the same scaling throughout this example
v = np.percentile(diff, [1, 99])

# Display difference
fig = mdreg.plot.par(
    diff, 
    title='Difference without coregistration', 
    vmin=v[0], 
    vmax=v[1],
)

#%%
# Coregister with elastix
# -----------------------
# We first use elastix to coregister the images:

t = time.time()

coreg, deform = mdreg.elastix.coreg(
    moving, 
    fixed, 
    spacing=spacing,
    FinalGridSpacingInPhysicalUnits=50.0,
)

print(f"elastix computation time: {round(time.time()-t)} seconds.")

# %%
# We used here the default b-spline registration method, but since this is 
# abdominal motion we used a coarser grid spacing than the 
# elastix default of 16mm (note: you can use `mdreg.elastix.defaults` to find 
# out what the defaults are). Any other defaults can be overridden by 
# specifying additional keywords.


#%% 
# We check the result by plotting the difference with the coregistered 
# (deformed) moving image:

fig = mdreg.plot.par(
    fixed - coreg, 
    title='Difference with elastix coregistration', 
    vmin=v[0], 
    vmax=v[1],
)

#%%
# The results clearly show the effect of the registration: the white line 
# at the top of the liver is gone but the gallbladder is deformed in an 
# unphysical way.

#%%
# Apart from the coregistered image, the function also returned the 
# transformation parameters. These can be used to deform other images in the 
# same way. As an example, we can check that transforming the moving image does 
# indeed produce the coregistered image:

# Deform the moving image
deformed = mdreg.elastix.transform(moving, deform, spacing)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")


#%%
# Coregister with skimage
# -----------------------
# We could try to improve the elastix coregistration by modifying the 
# parameters, but for the purpose of this tutorial we try another package 
# instead.
# 
# `skimage` has an implementation of the optical flow method for registration 
# which is wrapped by mdreg with the same API as elastix and ants. Let's 
# try it on our problem:

t = time.time()

coreg, deform = mdreg.skimage.coreg(
  moving, 
  fixed, 
  attachment=30.0,
)

print(f"skimage computation time: {round(time.time()-t)} seconds.")

# %%
# We chose to  
# use a coarser registration than the default by setting the attachment to a 
# higher value - 30 instead of the default 15 (note: as in elastix you can 
# find the default settings by calling `mdreg.skimage.defaults`).

#%% 
# Plot the difference with the coregistered (deformed) moving image:

fig = mdreg.plot.par(
    fixed - coreg, 
    title='Difference with skimage coregistration', 
    vmin=v[0], 
    vmax=v[1],
)

#%%
# This appears to have done a reasonable job at minimizing the difference 
# between the images without creating unwanted deformations. 

# %%
# In `skimage.coreg` the second return value is the deformation field. As in elastix 
# we can use it to deform other images in the same way. If we try this on the 
# moving image, we get the coregistered image again:
deformed = mdreg.skimage.transform(moving, deform)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")

#%%
# Coregister with ants
# --------------------
# Let's run this a final time with the third package wrapped in mdreg -
# ``ants``:

t = time.time()

coreg, deform = mdreg.ants.coreg(
    moving, 
    fixed,
    type_of_transform='SyNOnly',
)

print(f"ANTs computation time: {round(time.time()-t)} seconds.")

## %%
# We have used default settings for all parameters except the type 
# of transform. By default the transform is a two-stage process with affine 
# pre-alignment followed by deformable registration. Here we tried deformable 
# registration alone, which is more similar to what we have done with elastix.

#%% 
# Plot the difference with the coregistered (deformed) moving image:

fig = mdreg.plot.par(
    fixed - coreg, 
    title='Difference with ANTs coregistration', 
    vmin=v[0], 
    vmax=v[1],
)

#%%
# This also appears to have achieved the goal of reducing the main differences 
# without creating unwanted deformations.

# %%
# The second return value is a path or list of paths to files 
# that hold the parameter values. These can be used to deform other images 
# in the same way. If we deform the moving image, we get the coregistered 
# image again:

deformed = mdreg.ants.transform(moving, deform)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")

# %%
# Note since ants writes deformation parameters to files, this will leave 
# traces on disk unless you remove these explicitly:

[os.remove(d) for d in deform]

#%%
# Alternatively, if the transformation is not needed the coreg function can be 
# called with return_transfo=False:

coreg = mdreg.ants.coreg(
    moving, 
    fixed,
    type_of_transform='SyNOnly',
    return_transfo=False,
)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore