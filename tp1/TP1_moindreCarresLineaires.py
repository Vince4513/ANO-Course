import autograd as ag
import scipy.linalg as nla
import autograd.numpy as np
import matplotlib.pyplot as plt


###### 1. Moindres carrés linéaires ######

# Paramètres
alpha, beta, gamma, mu = 0.5,-2,1,7
m = 4

def f(x):
    return alpha*x**3 + beta*x**2 + gamma*x + mu

Tx = np.array([-1.1, 0.17, 1.22, -.5, 2.02, 1.81])
p = Tx.shape[0]
Ty_courbe = np.array([f(x) for x in Tx])
perturbations = 0.5*np.array([-1.3, 2.7, -5, 0, 1.4, 6])
Ty_exp = Ty_courbe + perturbations

erreur = Ty_courbe - Ty_exp
err_initiale = np.linalg.norm(erreur,2)
print(err_initiale)

xplot = np.linspace(-1.2,2.1,50)
yplot = np.array([f(x) for x in xplot])
plt.plot(xplot,yplot)

plt.scatter(Tx, Ty_exp)
plt.xlim(-1.2,2.1)
plt.show()

#Construction système d equations
A = np.empty ([p, m], dtype=np.float64)
b = Ty_exp
for j in range (0,m) :
    for i in range (0,p) :
        A[i,j] = Tx[i]**(m-1-j)
        
# Méthode historique
ATA = np.dot (np.transpose(A), A)
ATb = np.dot (np.transpose(A), b)
alpha, beta, gamma, mu = nla.solve(ATA, ATb)


#Affichage des paramètres et de l'erreur
erreur_minimale = nla.norm (Ty_courbe - np.array ([f(x) for x in Tx]), 2)

print("erreur initiale : ", err_initiale)
print("erreur minimale : ", erreur_minimale)

print("alpha  : ", alpha)
print("beta   : ", beta)
print("gamma  : ", gamma)
print("mu     : ", mu)

#Réaffichage avec la nouvelle courbe
xplot = np.linspace(-1.2,2.1,50)
yplot2 = np.array([f(x) for x in xplot])
plt.plot(xplot,yplot)
plt.plot(xplot,yplot2,"orange")
plt.scatter(Tx, Ty_exp)
plt.xlim(-1.2,2.1)
plt.show()



