#!/bin/python3

import autograd as ag
import autograd.numpy as np
import scipy.linalg as nla
import matplotlib.pyplot as plt

def f (a,b):
    return b * a**3 - 3*(b - 1) * a**2 + b**2 -1

def g (a,b):
    return a**2 * b**2 - 2

fig = plt.figure(figsize = (10,20))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.set_xlabel('$a$', labelpad=20)
ax.set_ylabel('$b$', labelpad=20)
ax.set_zlabel('$f(a,b)$ et $g(a,b)$', labelpad=20)

xplot = np.arange (-4, 4, 0.5)
yplot = np.arange (-4, 4, 0.5)

##### Graphe de f #####
X, Y = np.meshgrid (xplot, yplot)
Z = f(X,Y)

ax.plot_surface(X, Y, Z, cmap="spring_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 10, colors="k", linestyles="dashed")
ax.contour(X, Y, Z, 1, colors="red",  levels=np.array([0], dtype=np.float64), linestyles="solid")

##### Graphe de g #####
X, Y = np.meshgrid (xplot, yplot)
Z = g(X,Y)

ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X, Y, Z, 10, colors="k", linestyles="dashed")
ax.contour(X, Y, Z, 1, colors="green",  levels=np.array([0], dtype=np.float64), linestyles="solid")

plt.show()


## Question 2
def Jac_F (a,b):
    df_da = 3*b * a**2 - 6*a*(b-1)
    df_db = a**3 -3 * a**2+ 2*b
    dg_da = 2*a * b**2
    dg_db = 2*b * a**2
    
    return np.array( [ [df_da, df_db],
                       [dg_da, dg_db] ] )


## Question 3
def F (a,b):
    return np.array([f(a,b), g(a,b)])

#a, b = -0.5, 1.8
a, b = -3.0, 0.5
un = np.empty([10,2])
un[0,0] = a
un[0,1] = b
for i in range(0, 9):
    ha, hb = nla.solve(Jac_F(un[i,0], un[i,1]), -F(un[i,0],un[i,1]))
    un[i+1,0] = un[i,0] + ha
    un[i+1,1] = un[i,1] + hb
    
print("a : ", un[9,0])
print("b : ", un[9,1])


## Question 4
def F_autograd(u):
    a = u[0]
    b = u[1]
    return np.array([f(a,b), g(a,b)], dtype=np.float64)


jacobienne = ag.jacobian(F_autograd)

a, b = -0.5, 1.8
un = np.array([a,b], dtype=np.float64)
for i in range(0, 9):
    #print('u[%d] = ' % i, un)
    
    F = F_autograd(un)
    #print('F[%d] = ' % i, F, '\n')
    
    J = jacobienne(un)
    #print('J[%d] = \n' % i, J)
        
    h = nla.solve(J, -F)
    un = un + h
        
print("\nQ4 - un : ", un)


## Question 5
def F_forward(u):
    a = u[0]
    b = u[1]
    # Pour f(a,b)
    t1 = a ** 2
    t2 = b ** 2
    t3 = (a*b - 3*b + 3)*t1 + t2 - 1
    # Pour g(a,b)
    t1 = a ** 2
    t2 = b ** 2
    t4 = t1 * t2 - 2
    # Résultat
    return np.array ([t3, t4])

def nabla_Fforward(u) :
    a = u[0]
    b = u[1]
    t1 = a ** 2
    dt1_da = 2*a
    dt1_db = 0
    
    t2 = b ** 2
    dt2_da = 0
    dt2_db = 2*b
    
    t3 = (a*b - 3*b + 3)*t1 + t2 - 1
    dt3_da = b*t1 + a*b*dt1_da + dt2_da
    dt3_db = (a-3)*t1 + (a-3)*b*dt1_db + dt2_db
    
    t4 = t1 * t2 - 2
    dt4_da = dt1_da * dt2_da
    dt4_db = dt1_db * dt2_db
    
    return np.array([[dt3_da,dt3_db],
                     [dt4_da,dt4_db]],
                     dtype=np.float64)

jacobienne = ag.jacobian(F_forward)

a, b = -0.5, 1.8
un = np.array([a,b], dtype=np.float64)
for i in range(0, 9):
    #print('u[%d] = ' % i, un)
    
    F = F_forward(un)
    #print('F[%d] = ' % i, F, '\n')
    
    J = jacobienne(un)
    #print('J[%d] = \n' % i, J)
        
    h = nla.solve(J, -F)
    un = un + h
        
print("\nQ5 - un : ", un)