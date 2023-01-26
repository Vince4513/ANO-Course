#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:17:20 2022

@author: vreau
"""
import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt

## Question 9
Tx = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.float64)
Ty = np.array([0.53, 0.53, 1.53, 2.53, 12.53, 21.53, 24.53, 28.53, 28.53, 30.53], dtype=np.float64)

def modele(x, u):
    kappa = u[0]
    alpha = u[1]
    rho = u[2]
    return kappa / (1+np.exp(alpha - rho*x))

def logit(p):
    return np.log(p/(1-p))

def f (a,b):
    return a**3 + 2*a**2 - 2*a*b + b**2 + a - 2*b + 5

def nabla_f (a,b) :
    return np.array ([3*a**2+4*a-2*b+1, -2*a+2*b-2], dtype=np.float64)

# Contrôle du pas
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

def f1(u):
    s = 0
    for i in range (0, Tx.shape[0]):
        s += ((Ty[i]) - modele(Tx[i], u))**2
    return s

kappa, alpha, rho = 30.54, 5.163, 1.188
un = np.array([kappa, alpha, rho], dtype=np.float64)

#Affichage courbe regression logistique
plt.plot(Tx , modele(Tx, un))
plt.scatter(Tx,Ty)
plt.text(0,25,"Erreur modèle: "+str(round(f1(un),4)))
plt.legend(["Moindre carrés linéaires","Valeurs"])
plt.show()

print ('------------ Newton avec controle ------------')
for i in range (6) :
    a = un[1]
    b = un[2]
    h = nabla_f(a,b)
    z = f1(un)
    alpha = backtracking_line_search (un, 1., h)
    print ('u[%d] =' % i, un, 'f(u[%d]) =' % i, z, 'alpha =', alpha)
    un = un + alpha*h

print ('------------------ Gradient ------------------')
u = np.array([kappa, alpha, rho], dtype=np.float64)
for i in range (6) :
    a = un[1]
    b = un[2]
    z = f(a,b)
    h = - nabla_f (a,b)
    da = h[0]
    db = h[1]
    dz = - np.dot (h, h)
    gamma = np.sqrt (da**2 + db**2 + dz**2) * 10
    alpha = backtracking_line_search (un, .75, h)
    print ('u[%d] =' % i, un, 'f(u[%d]) =' % i, z, 'alpha =', alpha)
    un = un + alpha*h
