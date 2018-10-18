from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
nbins = 100
c = ['#2166ac', '#f4a582']
max_h = 0

cmap = LinearSegmentedColormap.from_list("mycmap", [c[0], c[1]], N=100)
z_array = np.arange(1., 11.)
cc = [cmap(i) for i in np.linspace(0, 1, len(z_array))]
for i, z in enumerate(z_array):
    ys = np.random.normal(loc=0, scale=z_array[-1-i], size=2000)

    hist, bins = np.histogram(ys, bins=nbins, density=True)
    xs = (bins[:-1] + bins[1:])/2
    width = (bins[1] - bins[0])

    max_h = max(np.append(hist, max_h))


    ax.bar(xs, hist, zs=z, zdir='y', color=cc[i], edgecolor="black", alpha=1., linewidth=0, width=width)

ax.set_xlabel('Distribution')
ax.set_ylabel('Epoch')
# ax.set_zlabel('Z')
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.set_yticks(np.arange(0, max(z_array), 5))
ax.view_init(45, 90)

# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(False)
# ax.set_zlim([0, max_h])

plt.show()