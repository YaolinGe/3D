import numpy as np
from matplotlib import pyplot as plt

d = 1

def grid_2d(xlen, ylen, d = 1):
    xx = np.arange(xlen) * d
    yy = np.arange(ylen) * d
    xgrid, ygrid = np.meshgrid(xx, yy)
    return xgrid, ygrid

xlen = 10
ylen = 10
xgrid, ygrid = grid_2d(xlen, ylen)

plt.figure()
plt.scatter(xgrid, ygrid)
plt.show()

print("hello world")

#%% hexagonal grid
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
phi = (np.sqrt(5)+1)/2
fig_width = 21
# %load_ext autoreload
# %autoreload 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
N = 121
Nx = int(np.sqrt(N))
Ny = N // Nx
xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse = False, indexing = 'xy')

fig, ax = plt.subplots(figsize = (fig_width, fig_width))
ax.scatter(xv, yv)
plt.show()

ratio = np.sqrt(3) / 2 # sin(60)
Nx = int(np.sqrt(N) / ratio)
Ny = N // Nx
xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse = False, indexing = 'xy')
fig, ax = plt.subplots(figsize = (fig_width, fig_width))
ax.scatter(xv, yv)
plt.show()

xv = xv * ratio
xv, yv

xv[::2, :] += ratio/2
fig, ax = plt.subplots(figsize=(fig_width, fig_width))
ax.scatter(xv, yv)


#%% test of new grid generation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

nx = 20.0
ny = 20.0
dx = 2
dy = dx * np.sin(np.pi / 3) # find the dy distance

xv, yv = np.meshgrid(np.arange(0, nx, dx), np.arange(0, ny, dy))
xv[::2, :] = xv[::2, :] + dx / 2 # shift x axis towards right by dx/2
zv = np.ones_like(xv)

fig = plt.figure(figsize = (10, 10))
# gs = GridSpec(nrows=1, ncols=1)
# ax0 = fig.add_subplot(gs[0])
# ax0 = fig.add_subplot()
ax0 = fig.add_subplot(111, projection = '3d')
# ax0 = Axes3D(fig)
ax0.scatter(xv, yv, zv, c = "black")
# ax0.scatter(xv, yv, c = "black")
ax0.set(xlabel = 'easting', ylabel = 'northing')
# plt.show()
#%%
# generate another layer's grid
xv_2, yv_2 = xv, yv
zv_2 = zv + 0.5
xv_2 = xv_2 + dx / 2
yv_2 = yv_2 + dx / 2 * np.tan(np.pi/6)
zv_2 = zv_2
# ax0.scatter(xv_2, yv_2, alpha = 0.4, c = "red")
ax0.scatter(xv_2, yv_2, zv_2, alpha = 0.4, c = "red")

xv_3, yv_3 = xv, yv
zv_3 = zv + 0.5
# zv_3 = zv_3
# xv_3 = xv_3
yv_3 = yv_3 + dy - dx / 2 * np.tan(np.pi/6)
ax0.scatter(xv_3, yv_3, zv_3, alpha = 0.4, c = "red")
# ax0.scatter(xv_3, yv_3, alpha = 0.4, c = "red")
plt.show()










