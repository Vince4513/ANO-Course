#!/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a - 2*b + 5

fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$', labelpad=20)

xmin, xmax = -1, 1.5
ymin, ymax = 0, 2.5
xplot = np.arange (xmin, xmax, 0.05)
yplot = np.arange (ymin, ymax, 0.05)

X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

m, n = Z.shape
for i in range (m) :
    for j in range (n) :
        Z[i,j] = max(min(Z[i,j], 8), 3)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 10, colors="k", linestyles="dotted")

x, y = -.2, 1.
ax.scatter (x, y, f(x,y), color='blue')
etiq = '$\\mathbf{u_0}$'
ax.text (x, y, f(x,y)+.1, etiq)
alpha = 2/3
ax.quiver (x, y, f(x,y), alpha*8/5, alpha*7/5, alpha*(-266/125), color='red')
ax.text (x + alpha*8/5, y + alpha*7/5, f(x,y) + alpha*(-266/125) + .5, '$D_h(f)(u_0)$')
alpha = 1/3
ax.quiver (x, y, 3, alpha*8/5, alpha*7/5, 0, color='black')
ax.text (x + 1/2*alpha*8/5, y + 1/2*alpha*7/5, 2.7, '$h$')
x, y = 7/5, 12/5
ax.scatter (x, y, f(x,y), color='blue')
etiq = '$\\mathbf{u_1}$'
ax.text (x, y, f(x,y)+.1, etiq)
x, y = 1/3, 4/3
ax.scatter (x, y, f(x,y), color='black')

curve_xplot = np.arange (xmin, xmax, 0.05)
def coord_y (x) :
    return (7/8)*x + 47/40

curve_yplot = coord_y (curve_xplot)
curve_zplot = f(curve_xplot, curve_yplot)
for i in range (curve_zplot.shape[0]) :
    curve_zplot[i] = max (min (curve_zplot[i], 8), 3)

ax.plot (curve_xplot, curve_yplot, curve_zplot)

x = -1
y = coord_y(x)
etiq = '$\\mathscr{C}$'
ax.text (x-.1, y+.2, f(x,y), etiq, fontsize=16)

plane_xplot = np.arange (xmin, xmax, 0.05)
plane_zplot = np.arange (2, 8, 0.05)

X, Z = np.meshgrid (plane_xplot, plane_zplot)
m, n = Z.shape
for i in range (m) :
    for j in range (n) :
        Z[i,j] = max(min(Z[i,j], 8), 3)

Y = coord_y (X)
ax.plot_surface(X, Y, Z, lw=0.5, rstride=1, cstride=1, alpha=0.5)

plt.show ()

