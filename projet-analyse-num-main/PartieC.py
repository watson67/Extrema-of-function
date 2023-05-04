#%%

from turtle import color
from matplotlib import colors
from matplotlib.pyplot import title
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         C.   Optimisation d'une fonction de deux variables réelles
#------------------------------------------------------------------------------

print("C.   Optimisation d'une fonction de deux variables réelles")

#-------------------------------- Question 1  ---------------------------------
print("Question 1")

def g_ab(x,y,a,b):
    return (x**2)/a + (y**2)/b

def g_227(x,y):
    return (x**2)/2 + (y**2)/(2/7)

def g_120(x,y):
    return (x**2)/1 + (y**2)/20

def h(x,y):
    return np.cos(x) * np.sin(y)
    
#Graph de g_ab
def plot_3D_gab(a,b):
    plt.ion()
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = g_ab(X,Y,a,b)
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none')
    ax.set_title("g(x,y)_2,2/7", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)

#Graph d'une fonction (ici h)
def plot_3D(func):
    x = np.linspace(-10, 10,200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    fig2 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none')
    ax.set_title("h(x,y)", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
plot_3D_gab(2,2/7)
plot_3D(h)
print()

#-------------------------------- Question 2  ---------------------------------

print("Question 2")

def plot_3D_LN_gab(a,b):
    plt.ion()
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = g_ab(X,Y,a,b)
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none', alpha=0.4)
    ax.set_title("contour g(x,y)_2,2/7", fontsize = 13)
    ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
def plot_3D_LN(func):
    x = np.linspace(-10, 10,200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    fig2 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none', alpha=0.4)
    ax.set_title("contour h(x,y)", fontsize = 13)
    ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)    

def plot_2D_LN(func):
    f = np.vectorize(func)
    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    fig3 = plt.figure(figsize = (10,10))
    plt.axis('equal')
    plt.contour(X, Y, Z, 15)
    plt.title("contour 2D h(x,y)", fontsize = 13)
    plt.xlabel('x', fontsize = 11)
    plt.ylabel('y', fontsize = 11)
    plt.show()

def plot_2D_LN_gab(a,b):
    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = g_ab(X, Y, a, b)
    fig4 = plt.figure(figsize = (10,10))
    plt.axis('equal')
    plt.contour(X, Y, Z, 15)
    plt.title("contour 2D gab(x,y)", fontsize = 13)
    plt.xlabel('x', fontsize = 11)
    plt.ylabel('y', fontsize = 11)
    plt.show()

# tracé des lignes de contours 2D et 3D des fonctions h et g_ab
plot_3D_LN_gab(2,2/7)
plot_3D_LN(h)
plot_2D_LN_gab(2,2/7)
plot_2D_LN(h)

#-------------------------------- Question 3  ---------------------------------

def grad_g_ab(a,b,x,y):
    grad = np.zeros(2)
    grad[0] = dg_ab_dx(a,b,x,y) #dg_ab/dx
    grad[1] = dg_ab_dy(a,b,x,y) #dg_ab/dy
    return grad

def dg_ab_dx(a,b,x,y):
    return (2 * x) / a

def dg_227_dx(x,y):
    return x

def dg_227_dy(x,y):
    return 7 * y 


def dg_ab_dy(a,b,x,y):
    return(2 * y) / b 

def grad_h(x,y):
    grad = np.zeros(2)
    grad[0] = dh_dx(x,y) #dh/dx
    grad[1] = dh_dy(x,y) #dh/dy
    return grad

def dh_dx(x,y):
    return - np.sin(x) * np.sin(y)

def dh_dy(x,y):
    return np.cos(x) * np.cos(y)



#-------------------------------- Question 4  ---------------------------------

def norme_gradient(tab):
    return np.sqrt(tab[0]**2 + tab[1]**2)

tab_2d = [
    [  0 ,  0 ],
    [ 7 ,  1.5],
    [ 10 , 10 ],
    [-10 , -10],
    [-54 , 20 ],
    [42  , 58 ],
    [100 , -26]
]

def q6():
    print("Calculs du gradient de h(x,y) et de sa norme  en quelques points :")
    for i in tab_2d:
        grad = grad_h(i[0],i[1])
        print("x = ", str(i[0]) + "; y = "+ str(i[1]))
        print("gradient = ")
        print(grad)
        print("Sa norme est  : " + str(norme_gradient(grad)))
        print()
        
    print("Calculs du gradient g_2,2/7(x,y) et de sa norme de en quelques points :")
    for i in tab_2d:
        grad = grad_g_ab(2,2/7,i[0],i[1])
        print("x = ", str(i[0]) + "; y = "+ str(i[1]))
        print("gradient = ")
        print(grad)
        print("Sa norme est  : " + str(norme_gradient(grad)))
        print()
        

#-------------------------------- Question 5  ---------------------------------

print("Question 5")

X = []
Y = []

def gradpc(eps, m, u, x0, y0, df1, df2,s):
    X = []
    Y = []
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration < m) :
        X.append(point[0])
        Y.append(point[1])
        point = point + u * grad
        grad[0] = df1(point[0],point[1])
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    fig5 = plt.figure(figsize = (15,10))
    plt.plot(X,Y)
    plt.title("descente de gradient de " + s, fontsize = 13)
    plt.legend([s])
    plt.xlabel('x', fontsize = 11)
    plt.ylabel('y', fontsize = 11)
    return point

def gradpc2(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration < m) :
        point = point + u * grad
        grad[0] = df1(point[0],point[1])
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    return f(point[0],point[1])

# g = gradpc(0.001,100,-0.1,0,0,dg_227_dx,dg_227_dy, "g_2,2/7")
# print(g)
# print(g_227(g[0],g[1]))
# print()

#-------------------------------- Question 6  ---------------------------------
N = 100
print("Question 6")

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
p = gradpc(0.0001,100,-0.1,0,0,dh_dx,dh_dy,"h(x,y)")
print(p)
print("h(x,y) = ")
print(h(p[0],p[1]))

print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
p = gradpc(0.0001,100,-0.1,7,1.5,dg_227_dx,dg_227_dy,"g_2,2/7")
print(p)
print(g_227(p[0],p[1]))

# tracé des courbes de niveau en 2D et 3D
def plot_XD_LN_gradpc(eps, m, u, x0, y0, df1, df2, s, func):
    XN = []
    YN = []
    X = []
    Y = []
    Z = []
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration < m) :
        XN.append(point[0])
        YN.append(point[1])
        point = point + u * grad
        grad[0] = df1(point[0],point[1])
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    # définition des points de la grille
    f = np.vectorize(func)
    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)
    Z = np.arange(len(XN)*len(YN)).reshape(len(XN),len(YN))
    X, Y = np.meshgrid(XN, YN)
    Z = f(X,Y)
    #tracé des courbes de niveau 2D
    fig10 = plt.figure(figsize = (10,2))
    plt.axis('equal')
    plt.contour(X, Y, Z, 15)
    plt.title(s + " 2D", fontsize = 13)
    plt.xlabel('Xn', fontsize = 11)
    plt.ylabel('Yn', fontsize = 11)
    plt.show()
    # tracé de courbes de niveau 3D
    fig20 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none', alpha=0.6)
    ax.set_title(s + " 3D", fontsize = 13)
    ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    ax.set_xlabel('Xn', fontsize = 11)
    ax.set_ylabel('Yn', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    ax.view_init(30, 120)

# Pour la fonction h tous les X du gradient sont égaux à 0, donc on ne peut pas tracer la fonction h sur une grille :
# plot_XD_LN_gradpc(0.0001,100,-0.1,0,0,dh_dx,dh_dy,"Contour gradient h(x,y)",h)
# En revanche on peux tracer la convergence de la fonction g227 sur une grille :
plot_XD_LN_gradpc(0.0001,100,-0.1,7,1.5,dg_227_dx,dg_227_dy,"Contour gradient g_2,2/7",g_227)
# Le minimum global de g_22/7 est obtenu pour le couple (0,0)

#-------------------------------- Question 7  ---------------------------------

def dg_120_dx(x,y):
    return 2*x

def dg_120_dy(x,y):
    return 2*y/10

def g_120(x,y):
    if x==0 and y==0 :
        return 10**-5 # on approxime 0 par epsilon = 10^-5
    else:
        return (x**2)/1 + (y**2)/20
        
ERR = []
X = np.zeros(120)
X = np.linspace(-0.99,-0.001,120) # pour u allant de -0.99 à -0.001 avec un pas automatique pour 120 valeurs
for i in range(120):
    ERR.append(abs(g_120(0,0) - gradpc2(0.0001,100,X[i],7,1.5,g_120,dg_120_dx,dg_120_dy))/g_120(0,0))
# tracé de l'erreur en fonction de u
fig11 = plt.figure(figsize = (15,10))
plt.plot(X,ERR, color = 'red')
plt.title("Erreur relative en focntion de u", fontsize = 13)
plt.xlabel('u', fontsize = 11)
plt.ylabel('Erreur', fontsize = 11)
plt.legend(["Erreur relative"])

#-------------------------------- Question 8  ---------------------------------

print("Question 8")
def F1(x,y,k,u,func, grad):
    x1 = x + k * u * grad[0]
    y1 = y + k * u * grad[1]
    return func(x1,y1)

def F2(x,y,k,u,func, grad):
    x1 = x + (k + 1) * u * grad[0]
    y1 = y + (k + 1) * u * grad[1]
    return func(x1,y1)

X = []
Y = []

def gradamax(eps, m, u, x0, y0, f, df1, df2,s):
    X = []
    Y = []
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        X.append(point[0])
        Y.append(point[1])
        k = 0
        f1 = 0 #on réinitialise f1 et f2, si on ne le fait pas, on ne rentrera qu'une fois dans le while suivant
        f2 = 1
        while(f1<f2):
            k +=1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
        fig6 = plt.figure(figsize = (15,10))
    plt.plot(X,Y, color = 'blue')
    plt.title("Recherche du maximum de " + s, fontsize = 13)
    plt.xlabel('X', fontsize = 11)
    plt.ylabel('Y', fontsize = 11)
    plt.legend([s])
    return point
    
print("Pour h(x,y) avec x0 = 2 et y0 = 0 :")
p = gradamax(0.001,100,0.1,2,0,h,dh_dx,dh_dy,"h(x,y)")
print(p)
print("h(x,y) = ")
print(h(p[0],p[1]))

# Ne converge pas, par conséquent on éxécute pas ces lignes :

# print("Pour g_2,2/7(x,y) avec x0 = 0 et y0 = 0 :")
# p = gradamax(0.001,100,0.1,7,1.5,g_227,dg_227_dx,dg_227_dy,"g_2,2/7")
# print(p)
# print("g_2,2/7(x,y) = ")
# print(g_227(p[0],p[1]))

#-------------------------------- Question 9  ---------------------------------

print("Question 9")

X = []
Y = []

def gradamin(eps, m, u, x0, y0, f, df1, df2,s):
    nb_iteration = 0
    X = []
    Y = []
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        X.append(point[0])
        Y.append(point[1])
        k = 0
        f1 = 1 #on réinitialise f1 et f2, si on ne le fait pas, on ne rentrera qu'une fois dans le while suivant
        f2 = 0
        while(f1>f2):
            k += 1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        nb_iteration += 1
    fig7 = plt.figure(figsize = (15,10))
    plt.plot(X,Y, color = 'blue')
    plt.title("Recherche du minimum de " + s, fontsize = 13)
    plt.xlabel('X', fontsize = 11)
    plt.ylabel('Y', fontsize = 11)
    plt.legend([s])
    return point

def gradamin2(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0)
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        k = 0
        f1 = 1 #on réinitialise f1 et f2, si on ne le fait pas, on ne rentrera qu'une fois dans le while suivant
        f2 = 0
        while(f1>f2):
            k += 1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        nb_iteration += 1
    return f(point[0],point[1])

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
p = gradamin(0.01,10,-0.1,0,0,h,dh_dx,dh_dy, "h(x,y)")
print(p)
print()
print("h(x,y) = ")
print(h(p[0],p[1]))
print()

print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
p = gradamin(0.0001,100,-0.1,7,1.5,g_227,dg_227_dx,dg_227_dy,"g_2;2/7")
print(p)
print()
print("g_2,2/7(x,y) = ")
print(g_227(p[0],p[1]))
print()

size = 30
errgradameliore = np.zeros(size)
for i in range(0,size):
    errgradameliore[i] = abs(gradamin2(0.0001,i+1,-0.1,7,1.5,g_227,dg_227_dx,dg_227_dy))

errgrad = np.zeros(size)
for i in range(0,size):
    errgrad[i] = abs(gradpc2(0.0001,i+1,-0.1,7,1.5,g_227,dg_227_dx,dg_227_dy))

x = [k for k in range(0,size)]
fig8 = plt.figure(figsize = (15,10))
plt.plot(x,errgradameliore, color = 'red')
plt.plot(x,errgrad, color = 'blue')
plt.title("Erreur de gradient", fontsize = 13)
plt.xlabel("Nombre d'itérations", fontsize = 11)
plt.ylabel("Erreur absolue", fontsize = 11)
plt.legend(["Gradient amélioré","Gradient"])

#-------------------------------- Question 10  ---------------------------------

print("Question 10")

# méthode permettant de récupérer le nombre d'itérations nécessaires pour atteindre le minimum via le gradient amélioré
def gradaminIT(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0)
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        k = 0
        f1 = 1 #on réinitialise f1 et f2, si on ne le fait pas, on ne rentrera qu'une fois dans le while suivant
        f2 = 0
        while(f1>f2):
            k += 1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        nb_iteration += 1
    return nb_iteration

# méthode permettant de récupérer le nombre d'itérations nécessaires pour atteindre le minimum via le gradiant
def gradpcIT(eps, m, u, x0, y0, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration < m) :
        point = point + u * grad
        grad[0] = df1(point[0],point[1])
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    return nb_iteration

X = np.zeros(120)
X = np.linspace(-0.99,-0.001,100) # pour u allant de -0.99 à -0.001 avec un pas automatique pour 100 valeurs
NBIGPC = []
NBIGMIN = []
for i in range(100):
    NBIGPC.append(gradpcIT(0.0001,100,X[i],1,20,dg_120_dx,dg_120_dy))
    NBIGMIN.append(gradaminIT(0.0001,100,X[i],1,20,g_120,dg_120_dx,dg_120_dy))

fig30 = plt.figure(figsize = (15,10))
plt.plot(X,NBIGPC, color = 'blue')
plt.plot(X,NBIGMIN, color = 'red')
plt.title("Nb d'itérations en fonction de u", fontsize = 13)
plt.xlabel("u", fontsize = 11)
plt.ylabel("Nombre d'itérations", fontsize = 11)
plt.legend(["Gradient","Gradient amélioré"])
# %%
