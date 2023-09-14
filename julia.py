import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

# Defining some constants, setting parameter for R for Julia set
c = complex(float(input("Please input Re(c).")),
            float(input("Please input Im(c).")))
print(c)
R = 0.5+(np.sqrt(1+4*abs(c))/2)
x_axis_upper = float(
    input("Please input a number for the upper bound of the real axis."))
x_axis_lower = float(
    input("Please input a number for the lower bound of the real axis."))
y_axis_upper = float(
    input("Please input a number for the upper bound of the imaginary axis."))
y_axis_lower = float(
    input("Please input a number for the lower bound of the imaginary axis."))
resolution = int(input(
    "Please input your resolution (how many pixels you want per integer increase on the real and imaginary axes)."))
# This sets the horizontal axis
sets = []
for i in range(round(resolution*x_axis_lower), round(resolution*x_axis_upper+1)):
    sets.append(i/resolution)
# This sets the vertical axis
sets2 = []
for i in range(round(resolution*y_axis_lower), round((resolution*y_axis_upper+1))):
    sets2.append(i/resolution)
complexs = []
for i in sets2:
    for j in sets:
        complexs.append(complex(j, i))

# Julia function


def julia(z, c, R):
    value = z**2+c
    N = 0
    while abs(value) <= R and N < 2500:
        value = value**2+c
        N += 1
    return N


# Uses Julia equation (z^2+c) to calculate the number of iterations needed to exceed R
field = []
for i in complexs:
    field.append(julia(i, c, R))
list1 = np.array(field)
# Splits list1 into x arrays

list2 = np.array(
    np.split(list1, round(resolution*(y_axis_upper-y_axis_lower)+1)))


dx, dy = 1/resolution, 1/resolution
# Horizontal and vertical axis setups
y, x = np.mgrid[slice(y_axis_lower, y_axis_upper+dy, dy),
                slice(x_axis_lower, x_axis_upper+dx, dx)]

z = list2
z = z[:-1, :-1]
levels = MaxNLocator(nbins=256).tick_values(z.min(), z.max())
cmap = plt.get_cmap('magma')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig, (ax1) = plt.subplots(nrows=1)
# im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
# fig.colorbar(im, ax=ax0, label='Iterations to R', fraction=0.1)
# ax0.set_title('Julia set for c='+str(c))
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')

cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(
    cf, ax=ax1, label='Iterations to R', fraction=0.1)
ax1.set_title('Julia set for c='+str(c))


# fig.tight_layout()

plt.show()
