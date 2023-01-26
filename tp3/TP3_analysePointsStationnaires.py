#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:10:57 2022

@author: vreau
"""

#!/bin/python3
import autograd as ag
import autograd.numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scipy.linalg as nla

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

## Question 2
def nabla_f(a,b):
    df_da = 3*a**2 + 4*a - 2*b + b**3
    df_db = -2*a + 2*b + 3*a * b**2 - 2
    return np.array([df_da, df_db], dtype=np.float64)

def H_f(a,b):
    d2f_da2 = 6*a + 4
    d2f_dadb = -2 + 3*b**2
    d2f_dbda = -2 + 3*b**2
    d2f_db2 = 2 + 6*a*b
    return np.array([
               [d2f_da2, d2f_dadb],
               [d2f_dbda, d2f_db2]], 
               dtype=np.float64
               )


## Question 3
a, b =  0.5, -2.0
a, b = -1.2, -0.3
a, b = -0.3,  1.6
a, b =  0.2,  1.0
a, b = -1.6, 0.8
un = np.empty([10,2])
un[0,0] = a
un[0,1] = b
for i in range(0, 9):
    ha, hb = np.linalg.solve(-H_f(un[i,0],un[i,1]),nabla_f(un[i,0],un[i,1]))
    un[i+1,0] = un[i,0] + ha
    un[i+1,1] = un[i,1] + hb

print("a : ", un[9,0])
print("b : ", un[9,1])


## Question 4
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

print("H definie positive ? : ", is_pos_def(H_f(un[9,0],un[9,1])))


## Question 5 
def fbis (u):
    a = u[0] 
    b = u[1]
    return a**3 + 2*a**2 - 2*a*b + b**2 + a * b**3 - 2*b + 5

def H_autograd(u):
    a = u[0] 
    b = u[1]
    d2f_da2 = 6*a + 4
    d2f_dadb = -2 + 3*b**2
    d2f_dbda = -2 + 3*b**2
    d2f_db2 = 2 + 6*a*b
    return np.array([
               [d2f_da2, d2f_dadb],
               [d2f_dbda, d2f_db2]], 
               dtype=np.float64
               )

grad_f = ag.grad(fbis)
hessienne = ag.hessian(fbis)

a, b = -1.5, 0.8
un = np.array([a,b], dtype=np.float64)
for i in range(0, 9):
    Nabla = grad_f(un)
    H = hessienne(un)
    
    h = nla.solve(-H, Nabla)
    un = un + h
        
print("\nQ5 - un : ", un)

## Question 6
def nabla_fter(u) :
    a = u[0]
    b = u[1]
    t1 = b ** 2
    t2 = a ** 2
    t3 = a * t1 * b - 2 * a * b + t2 * a - 2 * b + t1 + 2 * t2 + 5
    df_dt3 = 1
    df_dt2 = (a + 2) * df_dt3
    df_dt1 = (a*b + 1) * df_dt3
    df_da = 2 * df_dt2
    df_db = 2 * df_dt1
    return np.array([df_da,df_db], dtype=np.float64)

hessienne = ag.hessian(fbis)

a, b = -1.5, 0.8
un = np.array([a,b], dtype=np.float64)
for i in range(0, 9):
    N = nabla_fter(un)
    H = hessienne(un)
    
    h = np.linalg.solve(-H, N)
    un = un + h
        
print("\nQ6 - un : ", un)