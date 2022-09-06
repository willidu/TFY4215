# -*- coding: utf-8 -*-
#Python-program som finner energi-egenverdier og tilhÃ¸rende bÃ¸lgefunksjoner
#for en endimensjonal harmonisk oscillator. Potensialet er kvadratisk over
#en viss bredde pÃ¥ midten, og har en konstant verdi pÃ¥ hver side. 
#Partikkelen er et elektron. 
#Problemet lÃ¸ses som vanlig ved Ã¥ diskretisere den 2. deriverte. 
#Den resulterende Hamiltonmatrisen H
#diagonaliseres med bruk av numpy-funksjonen np.linalg.eigh(H), som regner ut
#og returnerer samtlige egenverdier og tilhÃ¸rende egenvektorer.
#Programmet plotter blant annet klassisk og kvantemekanisk sanns.tetthet.

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal

#m=elektronmassen, V0 = Npot eV
hbar=1.05E-34
m=9.11E-31
Npot=4
V0=Npot*1.6E-19
#N = 100 = antall posisjonsverdier i halvparten av det harmoniske omrÃ¥det
N=200
#dz = 1 Ã… = skrittlengden
dx=1E-10
#V = liste med potensialverdier 
V = [V0]*4*N + [V0*((n-N)/(N*1.0))**2 for n in range(2*N+1)] + [V0]*4*N
#regner ut vinkelfrekvensen omega
omega = np.sqrt(2.0*V[5*N+1]/m)/dx
print('omega=',omega)
#regner ut energisplittingen hbar*omega
ekvant = hbar*omega
ekvanteV = ekvant/1.6E-19
print('energikvant (eV)',ekvanteV)
#
Ntot=len(V)
V = np.asarray(V)
#d = liste med diagonalelementer i Hamiltonmatrisen H
d = [v + hbar**2/(m*dx**2) for v in V]
#e = liste med ikke-diagonale elementer i H, dvs H(i,i+-1) = e
e = [-hbar**2/(2*m*dx**2) for n in range(Ntot-1)]
#Finner w = egenverdiene og psi = egenvektorene til matrisen H
w,psi = eigh_tridiagonal(d,e)
x = np.asarray([dx*n for n in range(Ntot)])
xsym = x-Ntot*dx/2
xnm = x*1E9
xnmsym = xsym*1E9
#klassiske vendepunkter i nm:
A = 1E9*np.sqrt(2*w/m)/omega
#evalues = liste med egenverdier i enheten eV, med nullpunkt for
#energien i bunnen av det harmoniske potensialet:
evalues = w/1.6E-19
#Hvis Ã¸nskelig skriver neste linje ut de 4 laveste energiegenverdiene
print(evalues[0],evalues[1],evalues[2],evalues[3])
#Laber tabell VineV med potensialet i enheten eV
VineV = [pot/1.6E-19 for pot in V]

for i in range(3):
    plt.plot(xnmsym, psi.T[i], label=i)
plt.legend()
plt.show()
exit(1)

#Lager og plotter QM og klassisk sannsynlighetstetthet
#Velg en tilstand nr n
n=0
rhoqm = np.abs(psi[:,n]**2)/(dx*1E9)          #enhet 1/nm
rhokl = 1/(np.pi*np.sqrt(A[n]**2-xnmsym**2))  #enhet 1/nm
plt.plot(xnmsym,rhoqm,xnmsym,rhokl)
l = plt.axvline(x=-A[n], linewidth=1, color='k', ls = '--')
l = plt.axvline(x=A[n], linewidth=1, color='k', ls = '--')
plt.title('1D harmonisk oscillator',fontsize=20)
plt.xlabel('$x$ (nm)',fontsize=20)
plt.ylabel('$dP/dx$',fontsize=20)
plt.xlim(-1.2*A[n],1.2*A[n])
#plt.ylim(0.0,0.4)
plt.show()
print('Tilstand nr',n)
print('Numerisk energi (eV)',evalues[n])
print('Analytisk energi (eV)',hbar*omega*(n+0.5)/1.6E-19)
print('Max verdi for V(x) (eV)',np.max(VineV))