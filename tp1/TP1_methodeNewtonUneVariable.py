#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:39:08 2022

@author: vreau
"""
import autograd as ag
import scipy.linalg as nla
import autograd.numpy as np
import matplotlib.pyplot as plt


###### 2. Méthode de Newton ######$

##Question 2
# Fonctions
def f(x):
    return(x**3 -2)

def fprime(x):
    return(3*x**2)

v = [0,1,2,3,4,5]
fi = np.array([f(x) for x in v])


un = np.empty([10,2])
un[0,0] = 0
un[0,1] = 10
for i in range(0, 9):
    un[i+1,0] = i+1
    un[i+1,1] = un[i,1] - f(un[i,1])/fprime(un[i,1])

print("Premiers termes d'une suite qui tend vers racine cubique de 2 :\n",un)    

#xplot = np.linspace(-10,10,10)
#yplot = np.array([f(x) for x in xplot])
#plt.plot(xplot,yplot)
plt.scatter(un[:,0], un[:,1])
plt.show()

##Question 3
alpha, beta, gamma, mu = 1.5870106105525719, -3.1447447637103476, -0.37473299354311346, 7.484053576868045

def f_coef(x):
    return alpha*x**3 + beta*x**2 + gamma*x + mu

def fprime_coef(x):
    return 3*alpha*x**2 + 2*beta*x + gamma

def fseconde_coef(x):
    return 6*alpha*x + 2*beta

un = np.empty([10,2])
un[0,0] = 0
un[0,1] = 10
for i in range(0, 9):
    un[i+1,0] = i+1
    un[i+1,1] = un[i,1] - fprime_coef(un[i,1])/fseconde_coef(un[i,1])

print("Premiers termes d'une suite qui calcul le minimum local de la cubique précédente :\n",un)  


##Question 4
def g(u):
    x = u[0]
    return alpha*x**3 + beta*x**2 + gamma*x + mu

def gprime(u):
    x = u[0]
    return 3*alpha*x**2 + 2*beta*x + gamma

grad_f = ag.grad(g)
grad_fp = ag.grad(gprime)

k = 10
un = np.array([k], dtype=np.float64)
for i in range(0, 9):
    un = un - grad_f(un)/grad_fp(un)
    
print("\nQ4 - un : ", un)