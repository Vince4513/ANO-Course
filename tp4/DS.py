import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt

#Données
Tx = np.array([ 7, 12, 17.5, 20, 20, 23.5, 26, 30, 32, 34.5], dtype=np.float64)
Ty = np.array([10, 10,   20, 20, 34,   40, 34, 18,  9,   4 ], dtype=np.float64)

def modele(x,u):
    kappa, alpha, rho = u
    return kappa /(1+np.exp(alpha - rho*x))

def logit(p):
    return np.log(p/(1-p))

def modele_complet(x, u):
    kappa1, alpha1, rho1, mu, kappa2, alpha2, rho2 = u
    return kappa1 / (1+np.exp(alpha1 - rho1*x)) + mu - kappa2 / (1+np.exp(alpha2 - rho2*x))

def f1(u):
    s = 0
    for i in range (0, Tx.shape[0]):
        s += ((Ty[i]) - modele_complet(Tx[i], u))**2
    return s

def f2(u):
    s = 0
    for i in range (0, Tx.shape[0]):
        s += ((Ty[i]) - modele(Tx[i], u))**2
    return s

#Sigmoïde 1
Txc1 = Tx[0:6]
Tyc1 = Ty[0:6]
p1 = Txc1.shape[0]

kappa1, mu1 = 160, 10
m = 2
#Moindres carrés linéaires
# Construction système d equations
A = np.empty([p1, m], dtype=np.float64)
b = logit((Tyc1 - mu1)/kappa1)
for j in range(0, m):
    for i in range(0, p1):
        A[i, j] = Txc1[i] ** (m - 1 - j)

# Méthode historique
ATA = np.dot(np.transpose(A), A)
ATb = np.dot(np.transpose(A), b)
alpha1, rho1 = np.linalg.solve(ATA, ATb)
print ('alpha1: %f \n' % alpha1, 'rho1: %f \n' % rho1, 'A=', A)

u1 = np.array([kappa1, alpha1, rho1], dtype=np.float64)
#Affichage courbe regression logistique
plt.plot(Tx , modele(Tx, u1))
plt.scatter(Tx,Ty)
plt.text(7,35,"Erreur modèle: "+str(round(f2(u1),4)))
plt.legend(["Moindre carrés linéaires","Valeurs"])
plt.show()

#Sigmoïde 2
Txc2 = Tx[5:10]
Tyc2 = Ty[5:10]
p2 = Txc2.shape[0]

kappa2, mu2 = 176, 0
#Moindres carrés linéaires
# Construction système d equations
A = np.empty([p2, m], dtype=np.float64)
b = logit((Tyc2-mu2)/kappa2)
for j in range(0, m):
    for i in range(0, p2):
        A[i, j] = Txc2[i] ** (m - 1 - j)

# Méthode historique
ATA = np.dot(np.transpose(A), A)
ATb = np.dot(np.transpose(A), b)
alpha2, rho2 = np.linalg.solve(ATA, ATb)
print ('alpha2: %f \n' % alpha2, 'rho2: %f \n' % rho2, 'A=', A)

u2 = np.array([kappa2, alpha2, rho2], dtype=np.float64)
#Affichage courbe regression logistique
plt.plot(Tx , modele(Tx, u2))
plt.scatter(Tx,Ty)
plt.text(7,35,"Erreur modèle: "+str(round(f2(u2),4)))
plt.legend(["Moindre carrés linéaires","Valeurs"])
plt.show()


#Calcul de mu
mu = mu1 - mu2
print ('mu1: %f \n' % mu1, 'mu2: %f \n' % mu2, 'mu=', mu)

u = np.array([kappa1, alpha1, rho1, mu, kappa2, alpha2, rho2])
print(u)


#Affichage courbe regression logistique
plt.plot(Tx , modele_complet(Tx, u))
plt.scatter(Tx,Ty)
plt.text(7,35,"Erreur modèle: "+str(round(f1(u),4)))
plt.legend(["Moindre carrés linéaires","Valeurs"])
plt.show()