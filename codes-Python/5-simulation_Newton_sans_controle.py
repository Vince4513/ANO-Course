#!/bin/python3

import numpy as np

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a - 2*b + 5

def nabla_f (a,b) :
    return np.array ([3*a**2+4*a-2*b+1, -2*a+2*b-2], dtype=np.float64)

def H_f (a,b) :
    return np.array ([[6*a+4, -2], [-2, 2]], dtype=np.float64)

zero = np.array ([1/3, 4/3], dtype=np.float64)

a0 = -1/5
b0 = 1
un = np.array ([a0,b0], dtype=np.float64)
for i in range (6) :
    a = un[0]
    b = un[1]
    z = f(a,b)
    print ('u[%d] =' % i, un, 'f(u[%d]) =' % i, z)
    H_f_un = H_f (a,b)
    nabla_f_un = nabla_f (a,b)
    h = np.linalg.solve (- H_f_un, nabla_f_un)
    un = un + h

