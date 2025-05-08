import mdreg
import mdreg.plot as plt
#
# Get the data:
#
data = mdreg.fetch('MOLLI')
#
# Plot as animation:
#
plt.animation(data['array'], vmin=0, vmax=1e4)
