#!/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def f (x,y):
    return x**2 + 2*x*y - 1

def g (x,y):
    return x**2*y**2 - y - 1

def Jac (x,y) :
    return np.array ([[2*x+2*y, 2*x], [2*x*y**2, 2*x**2*y-1]], dtype=np.float64)

# Calculés avec Maple
abscisses = [-1.896238074, -0.4638513741, 1.568995999]
ordonnees = [0.6844390650, -0.8460057970, -0.4658228710]

x = -1.5
y = -.97

fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$', labelpad=20)

xplot = np.arange (-2, 1.7, 0.1)
yplot = np.arange (-1.5, 0.8, 0.1)

X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 10, colors="k", linestyles="dashed")
ax.contour(X, Y, Z, 1, colors="black",  levels=np.array([0], dtype=np.float64), linestyles="solid")

for i in range (len(abscisses)) :
    ax.scatter (abscisses[i], ordonnees[i], f(abscisses[i], ordonnees[i]), color='black')

#########################################################
# 2ème sous-graphe

ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$g(a,b)$', labelpad=20)

xplot = np.arange (-2, 1.7, 0.1)
yplot = np.arange (-1.5, 0.8, 0.1)

X, Y = np.meshgrid (xplot, yplot)
Z = g(X,Y)

ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 10, colors="k", linestyles="dashed")
ax.contour(X, Y, Z, 1, colors="black",  levels=np.array([0], dtype=np.float64), linestyles="solid")

for i in range (len(abscisses)) :
    ax.scatter (abscisses[i], ordonnees[i], g(abscisses[i], ordonnees[i]), color='black')

# plt.savefig ("../deux-variables-graphe.png")

plt.show()


