"""
==============================================
Groupwise coregistration API
==============================================

``mdreg`` includes a harmonized API for groupwise coregistration of 
series of 2D images or 3D volumes with three different packages: ``ants``, 
``skimage`` and ``itk-elastix``. This examples illustrates their usage.
"""

#%%
# Setup
# -----
import time
import numpy as np
import mdreg


#%%
# Load test data
data = mdreg.fetch('VFA')

# Select short series of 3 volumes to coregister
fixed = data['array'][:,:,:,:3]
moving = data['array'][:,:,:,-3:]

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
anim = mdreg.plot.animation(
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

coreg, deform = mdreg.elastix.coreg_series(
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

anim = mdreg.plot.animation(
    fixed - coreg, 
    title='Difference with elastix coregistration', 
    vmin=v[0],
    vmax=v[1],
)

#%%
# Apart from the coregistered image, the function also returned the 
# transformation parameters. These can be used to deform other images in the 
# same way. As an example, we can check that transforming the moving image does 
# indeed produce the coregistered image:

# Deform the moving image
deformed = mdreg.elastix.transform_series(moving, deform, spacing=spacing)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")


#%%
# Coregister with skimage
# -----------------------
# `skimage` has an implementation of the optical flow method for registration 
# which is wrapped by mdreg with the same API as elastix and ants. Let's 
# try it on our problem:

t = time.time()

coreg, deform = mdreg.skimage.coreg_series(
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

anim = mdreg.plot.animation(
    fixed - coreg, 
    title='Difference with skimage coregistration', 
    vmin=v[0],
    vmax=v[1],
)

# %%
# In `skimage.coreg_series` the second return value is the deformation field. 
# As in elastix 
# we can use it to deform other images in the same way. If we try this on the 
# moving image, we get the coregistered image again:
deformed = mdreg.skimage.transform_series(moving, deform)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")

#%%
# Coregister with ants
# --------------------
# Let's run this a final time with the third package wrapped in mdreg -
# ``ants``:

t = time.time()

coreg, deform = mdreg.ants.coreg_series(
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

anim = mdreg.plot.animation(
    fixed - coreg, 
    title='Difference with ANTs coregistration', 
    vmin=v[0],
    vmax=v[1],
)

#%%
# This also appears to have acheived the goal of reducing the main differences 
# without creating unwanted deformations.

# %%
# The second return value is a path or list of paths to files 
# that hold the parameter values. These can be used to deform other images 
# in the same way. If we deform the moving image, we get the coregistered 
# image again:

deformed = mdreg.ants.transform_series(moving, deform)

# Check the difference with the coregistered image
err = 100*np.linalg.norm(deformed-coreg)/np.linalg.norm(moving)

print(f"Difference between coregistered and deformed: {err} %")

# %%
# Note since ants writes deformation parameters to files, this will leave 
# traces on disk unless you remove these explicitly.

#%%
# Alternatively, if the transformation is not needed the coreg function can be 
# called with return_transfo=False:

coreg = mdreg.ants.coreg_series(
    moving, 
    fixed,
    type_of_transform='SyNOnly',
    return_transfo=False,
)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore