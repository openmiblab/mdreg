import mdreg
#
# Get the data:
#
data = mdreg.fetch_zarr('VFA')
#
# Plot as animation:
#
mdreg.plot.animation(data, vmin=0, vmax=1e4)
