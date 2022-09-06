# -*- coding: utf-8 -*-
#Partikkel i 1D boks

import matplotlib.pyplot as plt
import numpy as np

#m=masse (SI)
hbar=1.05E-34
m=9.11E-31  #elektron
V0=0.00
#N = antall diskrete posisjonsverdier i boksen
N=100
#dz = skrittlengde (m); boksbredde = N*dz
dz=0.040E-9
#V = potensialet i boksen (V=0)
V = [V0]*N
#d = liste med diagonalelementer i Hamiltonmatrisen H
d = [v + hbar**2/(m*dz**2) for v in V]
#e = verdi til ikke-diagonale elementer i H, dvs H(i,i+-1)
e = - hbar**2/(2*m*dz**2)
#Ntot = antall elementer i V
Ntot=len(V)
#Initialisering av matrisen H: Legger inn verdi 0 i samtlige elementer
H = [[0]*(Ntot) for n in range(Ntot)]
#Dobbel for-lÃ¸kke som lager den tridiagonale matrisen H
for i in range(Ntot):
    for j in range(Ntot):
        if i==j:
            H[i][j]=d[i]
        if abs(i-j)==1:
            H[i][j]=e
#Finner w = egenverdiene (J) og psi = egenvektorene til matrisen H           
w,psi = np.linalg.eigh(H)
#evalues = liste med energiegenverdier i enheten eV
evalues = w/1.60E-19
print(f'{w[0:4] = }')
print(f'{evalues[0:4] = }')
plt.plot(evalues)
plt.show()

#z = liste med posisjonsverdier (nm)
z = [dz*n*1E9 for n in range(Ntot)]
#Skriver ut de 6 laveste energiegenverdiene i enheten eV
#print(w[0],w[1],w[2],w[3],w[4],w[5])
# print(evalues[0],evalues[1],evalues[2],evalues[3],evalues[4],evalues[5])
#psi1 = bÃ¸lgefunksjonen til grunntilstanden, psi2 = 1. eksiterte tilstand osv
psi1 = psi[:,0]
psi2 = psi[:,1]
psi3 = psi[:,2]
psi4 = psi[:,3]
#Plotter bÃ¸lgefunksjonene som tilsvarer de 4 laveste egenverdiene
plt.figure('Wave functions')
plt.plot(z,psi1,z,psi2,z,psi3,z,psi4)
plt.title('Particle in 1D box',fontsize=20)
plt.xlabel('$z$ (nm)',fontsize=20)
plt.ylabel('$\psi$',fontsize=20)
# plt.show()
#Plotter potensialet og en strek for hver av de 4 laveste egenverdiene
#Samme fargerekkefÃ¸lge som for bÃ¸lgefunksjonene
#(b=blÃ¥, g=grÃ¸nn, r=rÃ¸d, c=cyan)
plt.figure('Energy levels')
plt.plot(z,V)
l = plt.axhline(y=evalues[0], linewidth=1, color='b')
l = plt.axhline(y=evalues[1], linewidth=1, color='g')
l = plt.axhline(y=evalues[2], linewidth=1, color='r')
l = plt.axhline(y=evalues[3], linewidth=1, color='c')
plt.title('Particle in 1D box',fontsize=20)
plt.xlabel('$z$ (nm)',fontsize=20)
plt.ylabel('$E$ (eV)',fontsize=20)
# plt.show()
#Plotter sannsynlighetstettheten for de 4 laveste tilstandene
plt.figure('Probability density')
plt.plot(z,np.abs(psi1**2),z,np.abs(psi2**2),z,np.abs(psi3**2),z,np.abs(psi4**2))
plt.title('Particle in 1D box',fontsize=20)
plt.xlabel('$z$ (nm)',fontsize=20)
plt.ylabel('$|\psi|^2$',fontsize=20)
plt.show()