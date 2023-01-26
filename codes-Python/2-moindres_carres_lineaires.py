#!/bin/python3

import numpy as np
import scipy.linalg as nla
import matplotlib.pyplot as plt

alpha = 3
beta = 4
gamma = -5

def f(x) :
    return alpha*x**2 + beta*x + gamma

m = 3
p = 4
Tx = np.array ([2,-1,4,3], dtype=np.float64)
Ty = np.array ([f(x) for x in Tx])
Ty = Ty + np.random.random (p) 

plt.scatter (Tx, Ty)
xplot = np.linspace (-1.1, 4.1, 50)
yplot = [f(x) for x in xplot]
plt.plot (xplot, yplot)
plt.show ()

erreur = nla.norm (Ty - np.array ([f(x) for x in Tx]), 2)

A = np.empty ([p, m], dtype=np.float64)
b = Ty
for j in range (0,m) :
    for i in range (0,p) :
        A[i,j] = Tx[i]**(2-j)

# Méthode historique

ATA = np.dot (np.transpose(A), A)
ATb = np.dot (np.transpose(A), b)

# Pour les paresseux
nla.solve(ATA, ATb)

# Pour les courageux
L = nla.cholesky(ATA, lower=True)

# Résolution en deux temps
y = nla.solve_triangular(L, ATb, lower=True)
# Pour les paresseux
nla.solve_triangular (np.transpose(L), y, lower=False)
# Pour les plus courageux
x = nla.solve_triangular (L, y, lower=True, trans='T')

alpha, beta, gamma = x
erreur_minimale = nla.norm (Ty - np.array ([f(x) for x in Tx]), 2)

# Méthode moderne

# Factorisation QR complète
Q, R = nla.qr(A)
# Factorisation réduite
Q1, R1 = nla.qr(A, mode='economic')
Q1Tb = np.dot (np.transpose (Q1), b)

nla.solve_triangular (R1, Q1Tb, lower=False)



