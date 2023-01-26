#!/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a*b**3 - 2*b + 5

def nabla_f (a,b) :
    return np.array ([b**3 + 3*a**2 + 4*a - 2*b, 
                      3*a*b**2 - 2*a + 2*b - 2], dtype=np.float64)

fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$', labelpad=20)

xplot = np.arange (-.8, 1.2, 0.05)
yplot = np.arange (-.8, 1.2, 0.05)

# Tracé du graphe de f
X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

m, n = Z.shape
for i in range (m) :
    for j in range (n) :
        Z[i,j] = max(min(Z[i,j], 12), 3)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 30, colors="k", linestyles="dotted")

# Le minimim local de f sans tenir compte de la contrainte
zero = np.array ([0.2255014396, 0.9318083312])
ax.scatter ([zero[0]], [zero[1]], [f(zero[0],zero[1])], color='black')

# Tracé de la contrainte a**2 + b**2 = r**2 (avec r**2 = 1/2)

r = 1/np.sqrt(2)
n = 100
angles = np.linspace (-np.pi, np.pi, n)
cxplot = np.array ([r * np.cos(theta) for theta in angles])
cyplot = np.array ([r * np.sin(theta) for theta in angles])
czplot = np.array ([f (cxplot[i], cyplot[i]) for i in range (0,n)])
ax.plot (cxplot, cyplot, czplot, color='black')

# Tracé d'un gradient (longueur normalisée)

a, b = 1, 1
grad_f = nabla_f (a, b)
grad_f = (.25/np.linalg.norm(grad_f,2)) * grad_f
ax.quiver (a, b, f(a,b), grad_f[0], grad_f[1], 0, color='blue')
ax.text (a, b, f(a,b), 'un gradient')
plt.show ()

