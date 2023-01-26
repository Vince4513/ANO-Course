#!/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import autograd as ag

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a * b**3 - 2*b + 5

fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$', labelpad=20)

xplot = np.arange (-5, 2, 0.1)
yplot = np.arange (-5, 3, 0.1)

X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

zmin = 3
zmax = 14

m, n = Z.shape
for i in range (m) :
    for j in range (n) :
        Z[i,j] = max(min(Z[i,j], zmax), zmin)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 30, colors="k", linestyles="dotted")

plt.show ()