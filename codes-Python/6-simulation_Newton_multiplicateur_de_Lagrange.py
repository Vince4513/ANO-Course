#!/bin/python3

import math
import numpy as np

def f (a,b) :
    return a**3 + 2*a**2 - 2*a*b + b**2 + a - 2*b + 5

def nabla_f (a,b) :
    return np.array ([3*a**2+4*a-2*b+1, -2*a+2*b-2])

def c (a,b) : 
    return a**2 + b**2 - 1

def nabla_c (a, b) :
    return np.array ([2*a, 2*b])

def Lagrangien (a,b,lmbda) :
    return f(a,b) + lmbda*c(a,b)

def nabla_Lagrangien (a,b,lmbda) :
    return np.array ([2*lmbda*a+3*a**2+4*a-2*b+1, 
                      2*lmbda*b-2*a+2*b-2, 
                      a**2+b**2-1])

def Hessian_Lagrangien (a,b,lmbda) :
    return np.array ([[2*lmbda+6*a+4,        -2, 2*a], 
                      [           -2, 2*lmbda+2, 2*b], 
                      [          2*a,       2*b, 0]])

print ('------------------ Newton ------------------')
# Celle-ci converge vers un autre point stationnaire
u = np.array ([-1.7, 1.7, 0])
# Celle-ci converge vers le minimum local
u = np.array ([0.4, 1.5, 0])
for i in range (6) :
    a = u[0]
    b = u[1]
    lmbda = u[2]
    z = Lagrangien (a, b, lmbda)
    print ('u[%d] =' % i, u, 'f(u[%d]) =' % i, z)
    H = Hessian_Lagrangien(a,b,lmbda)
    g = nabla_Lagrangien(a,b,lmbda)
    h = np.linalg.solve (- H, g)
    u = u + h
    grad_f = nabla_f (a,b)
    grad_c = nabla_c (a,b)
    grad_f = (1/np.linalg.norm(grad_f,2)) * grad_f
    grad_c = (1/np.linalg.norm(grad_c,2)) * grad_c
    print ('    nabla f normalisé =', grad_f)
    print ('    nabla c normalisé =', grad_c)
    print ('    valeur de la contrainte =', c(a,b))


