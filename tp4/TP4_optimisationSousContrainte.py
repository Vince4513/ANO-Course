#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:24:36 2022

@author: vreau
"""

#!/bin/python3
import math as mt
import autograd.numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import autograd as ag

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


## Question 1
def c (a,b) :
    return a**2 + b**2 - 1/2

## Question 3
def nabla_c (a,b) :
    return np.array ([2*a, 2*b], dtype=np.float64)

# Tracé d'un gradient (longueur normalisée)
angle = 0
for angle in np.linspace(0,10,10):
    a, b = r*np.cos(angle), r*np.sin(angle)
    
    grad_f = nabla_f (a, b)
    grad_f = (.25/np.linalg.norm(grad_f,2)) * grad_f
    ax.quiver (a, b, f(a,b), grad_f[0], grad_f[1], 0, color='blue')
    ax.text (a, b, f(a,b), 'grad_f')

    # Tracé d'un gradient de c(longueur normalisée)
    grad_c = nabla_c (a, b)
    grad_c = (.25/np.linalg.norm(grad_c,2)) * grad_c
    ax.quiver (a, b, f(a,b), grad_c[0], grad_c[1], 0, color='red')
    ax.text (a, b, f(a,b), 'grad_c')
    
plt.show ()

## Question 5
def Lagrangien (u) :
    a = u[0]
    b = u[1]
    lmbda = u[2]
    return f(a,b) + lmbda*c(a,b)


## Question 6
# =============================================================================
# def nabla_Lagrangien (u) :
#     a = u[0]
#     b = u[1]
#     lmbda = u[2]
#     return np.array ([2*lmbda*a+3*a**2+4*a-2*b+1,
#                       2*lmbda*b-2*a+2*b-2,
#                       a**2+b**2-0.5])
# 
# def H_Lagrangien (u) :
#     a = u[0]
#     b = u[1]
#     lmbda = u[2]
#     return np.array ([[2*lmbda+6*a+4,        -2, 2*a],
#                       [           -2, 2*lmbda+2, 2*b],
#                       [          2*a,       2*b,   0]])
# =============================================================================

nabla_Lagrangien = ag.grad(Lagrangien)
H_Lagrangien = ag.hessian(Lagrangien)

## Question 7
u = np.array ([0.17, 0.68, 0])
for i in range (6) :
    z = Lagrangien (u)
    print ('u[%d] =' % i, u, 'f(u[%d]) =' % i, z)
    H = H_Lagrangien(u)
    g = nabla_Lagrangien(u)
    h = np.linalg.solve (- H, g)
    u = u + h
    grad_f = nabla_f (a,b)
    grad_c = nabla_c (a,b)
    grad_f = (1/np.linalg.norm(grad_f,2)) * grad_f
    grad_c = (1/np.linalg.norm(grad_c,2)) * grad_c
    print ('nabla f normalisé =', grad_f)
    print ('nabla c normalisé =', grad_c)
    print ('valeur de la contrainte =', c(a,b))
    print ('\n')


## Question 8
Tx = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.float64)
Ty = np.array([], dtype=np.float64)

def g(u):
    alpha = u[0]
    rho = u[1]
    kappa = u[2]   

    res= 0.0
    N = Tx.shape[0]
    for i in range(N):
        res = res + (Ty[i] - kappa /(1 + np.exp(alpha-rho*Tx[i]))**2)
    
    return res
        
