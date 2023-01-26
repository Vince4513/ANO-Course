#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def f(a) :
    return np.sqrt(a**2 - a**3)

plt.figure()
plt.axis ('equal')
plt.xlabel ('$a$')
plt.ylabel ('$b$')
plt.axhline (color='black')

xplot = np.arange (-1.2, 1.001, .01)
xplot[-1] = 1.
yplot = np.array ([f(a) for a in xplot])
plt.plot (xplot, yplot, color='blue')
yplot = np.array ([-f(a) for a in xplot])
plt.plot (xplot, yplot, label='solutions de $a^3 - a^2 + b^2 = 0$', color='blue')

plt.legend ()
plt.show ()

