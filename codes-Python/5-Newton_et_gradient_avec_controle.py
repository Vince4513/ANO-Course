#!/bin/python3
  
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a - 2*b + 5

def nabla_f (a,b) :
    return np.array ([3*a**2+4*a-2*b+1, -2*a+2*b-2], dtype=np.float64)

def H_f (a,b) :
    return np.array ([[6*a+4, -2], [-2, 2]], dtype=np.float64)

fig = plt.figure(figsize = (20,20))
ax = plt.axes(projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$', labelpad=20)

xplot = np.arange (-.3, .5, 0.05)
yplot = np.arange ( .8, 1.6, 0.05)

X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

m, n = Z.shape
for i in range (m) :
    for j in range (n) :
        Z[i,j] = max(min(Z[i,j], 8), 3)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 30, colors="k", linestyles="dotted")

ax.scatter (1/3, 4/3, f(1/3,4/3), color='black')

def backtracking_line_search (un, alpha0, h) :
    rho = .5
    c = .5
    alpha = alpha0
    a = un[0]
    b = un[1]
    D_h = np.dot (nabla_f (a,b), h)
    while f(a + alpha*h[0], b + alpha*h[1]) > f(a,b) + c * alpha * D_h :
        alpha *= rho
    return alpha

a0 = -1/5
b0 = 1
un = np.array ([a0,b0], dtype=np.float64)
for i in range (4) :
    a = un[0]
    b = un[1]
    z = f(a,b)
    ax.scatter (a, b, z, color='blue')
    if i == 0 :
        etiq = '$\\mathbf{u_' + str(i) + ' = \\bar{u}_' + str(i) + '}$'
    else :
        etiq = '$\\mathbf{u_' + str(i) + '}$'
    if i < 3 :
        ax.text (a, b, z+.05, etiq)
    else :
        ax.text (a+.02, b, z+.05, etiq)
    H_f_un = H_f (a,b)
    nabla_f_un = nabla_f (a,b)
    h = np.linalg.solve (- H_f_un, nabla_f_un)
    alpha = backtracking_line_search (un, 1., h)
    un = un + alpha*h

un = np.array ([a0,b0], dtype=np.float64)
for i in range (6) :
    a = un[0]
    b = un[1]
    z = f(a,b)
    if i > 0 :
        ax.scatter (a, b, z, color='red')
    etiq = '$\\mathbf{\\bar{u}_' + str(i) + '}$'
    if i > 0 and i < 3 :
        ax.text (a, b, z+.05, etiq)
    elif i > 0 :
        ax.text (a+.02, b, z+.05, etiq)
    h = - nabla_f (a,b)
    da = h[0]
    db = h[1]
    dz = - np.dot (h, h)
    gamma = np.sqrt (da**2 + db**2 + dz**2) * 10
    ax.quiver (a, b, z, da/gamma, db/gamma, dz/gamma)
    alpha = backtracking_line_search (un, .75, h)
    un = un + alpha*h

plt.show ()
