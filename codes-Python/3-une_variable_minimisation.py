#!/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

alpha, beta, gamma, mu = 1.5870106105525719, \
    -3.1447447637103476, \
    -0.37473299354311346, \
    7.484053576868045

def f(x):
    return alpha*x**3 + beta*x**2 + gamma*x + mu

def fprime(x) :
    return 3*alpha*x**2 + 2*beta*x + gamma

def fseconde (x) :
    return 6*alpha*x + 2*beta

xplot = np.linspace (-1.2, 2.9, 100)

fig = plt.figure(figsize = (20,20))

ax_left = fig.add_subplot(1, 2, 1)

ax_left.set_xlabel('$x$', labelpad=20)
ax_left.set_ylabel('$f(x,y)$', labelpad=20)

yplot = np.array ([f(x) for x in xplot])

ax_left.plot (xplot, yplot, label='graphe de f')

ax_left.axhline (color='black')

ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)

ax_right.set_xlabel('$x$', labelpad=20)
ax_right.set_ylabel("$\\frac{\\mathrm{d}f}{\\mathrm{d}x}(x,y)$", labelpad=20)

yplot = np.array ([fprime(x) for x in xplot])

ax_right.axhline (color='black')

ax_right.plot (xplot, yplot, label='graphe de f')

u0 = 0.8
for i in range (3) :
    u1 = u0 - fprime(u0)/fseconde(u0)
    ax_left.scatter ([u0],[f(u0)])
    ax_left.plot ([u0,u1],[f(u0),0], linestyle='dotted')
    ax_left.plot ([u1,u1],[f(u1),0], linestyle='dotted', color='black')
    ax_right.scatter ([u0],[fprime(u0)])
    ax_right.plot ([u0,u1],[fprime(u0),0], linestyle='dotted')
    ax_right.plot ([u1,u1],[fprime(u1),0], linestyle='dotted', color='black')
    u0 = u1

ax_left.scatter ([u0],[f(u0)])
ax_right.scatter ([u0],[fprime(u0)])

plt.show ()
