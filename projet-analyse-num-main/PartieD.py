#%%

from turtle import color
from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         D.   Calcul de l'inverse d'une matrice
#------------------------------------------------------------------------------

#-------------------------------- Question 1  ---------------------------------

# Voir le rapport

#-------------------------------- Question 2  ---------------------------------

print("\nQuestion 2 : ")

# 9x + 0y = -3	-> 	A[0,0]x A[0,1]y = b[0] 
# 0x + 9y = 5	-> 	A[1,0]x A[1,1]y = b[1]
b = np.array([-6,9])			# résutats des équantions du système d'équations
A = np.array([[12,5],[5,12]])	# Matrice définie symmétrique positive de taille n

def G(y):
    return 2*(np.matmul(A,y)-b)

def rho(y):
	if np.array_equal(G(y),[0,0]):
		return 0
	else:
		return ((np.linalg.norm(G(y)))**2)/(2*np.matmul(np.transpose(G(y)),np.matmul(A,G(y))))

def solveur(A, b):          # A matrice carrée de taille n, b les n résutats des équantions du système d'équations
    m = 100
    n = len(A)
    eps = 10**-6
    Yn = 3*np.ones(n)       # Initialisation de Yn avec des valeurs arbitraires (ici 3 pour toutes les composantes)
    nb_itérations = 0
    Y = Yn - rho(Yn)*G(Yn)
    while (nb_itérations <= m and np.linalg.norm(Y-Yn) > eps):
        nb_itérations += 1
        Yn = Y
        Y = Yn - rho(Yn)*G(Yn)
    return Yn

#-------------------------------- Question 3  ---------------------------------

print("\nQuestion 3 : ")

print("\ncas n=2 : \n")

A = np.array([[12,5],[5,12]])
print("matrice A : \n", A)
b = np.array([-6,9])
print("array b : \n", b)

print("\nSolution : ")
print(solveur(A,b))

print("\ncas n=3 : \n")

A = np.array([[12,5,1],[5,12,1],[1,1,12]])
print("matrice A : \n", A)
b = np.array([-6,9,1])
print("array b : \n", b)

print("\nSolution : ")
print(solveur(A,b))
# %%
