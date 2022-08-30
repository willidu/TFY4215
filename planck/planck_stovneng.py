# -*- coding: utf-8 -*-
#TFY4215/TFY4165 2022
#Plancks strÃ¥lingslov

import matplotlib.pyplot as plt
import numpy as np

#Konstanter
c = 299792458 #m/s
h = 6.62607015E-34 #Js
kB = 1.380649E-23 #J/K

T1=3000 #K
E1=kB*T1 #J
f1=E1/h #Hz
L1=c/f1 #m
T2=1500 #K
E2=kB*T2 #J
f2=E2/h #Hz
L2=c/f2 #m
N=1000
f = np.zeros(N)
L = np.zeros(N)
djdf1 = np.zeros(N)
djdf2 = np.zeros(N)
fmax1=0
imax1=0
djdL1 = np.zeros(N)
djdL2 = np.zeros(N)
Lmax1=0
iLmax1=0

for i in range(3,N,1):
    f[i]=10*i*f1/N
    L[i]=1*i*L1/N
    djdf1[i]=2*np.pi*h*f[i]**3/(c*c*(np.exp(f[i]/f1)-1))
    djdf2[i]=2*np.pi*h*f[i]**3/(c*c*(np.exp(f[i]/f2)-1))
    djdL1[i]=(2*np.pi*h*c*c/L[i]**5)/(np.exp(L1/L[i])-1)
    djdL2[i]=(2*np.pi*h*c*c/L[i]**5)/(np.exp(L2/L[i])-1)
    if i>0:
        if djdf1[i]>djdf1[i-1]:
            fmax1=f[i]
            imax1=i
        if djdL1[i]>djdL1[i-1]:
            Lmax1=L[i]
            iLmax1=i
    if i==0:
        djdf1[i]=0
        djdf2[i]=0
        djdL1[i]=0
        djdL2[i]=0

print('fmax1/T1=',fmax1/T1,'Hz/K')
print('Lmax1*T1=',Lmax1*T1,'m*K')
#Plotting
plt.figure('Planckkurver')
plt.plot(f*1E-12,djdf1*1E9,label='3000 K')
plt.plot(f*1E-12,djdf2*1E9,label='1500 K')
plt.legend(loc='best')
plt.title('Planckkurver',fontsize=20)
plt.xlabel('f (THz)',fontsize=24)
plt.ylabel('dj/df (nW/m^2 Hz)',fontsize=24)
#plt.savefig("planckkurver.eps")
plt.grid()
plt.show()
#np.set_printoptions(precision=2)
#print('j(3000)/j(1500)=',np.sum(djdf1)/np.sum(djdf2))
#fblue=c/400E-9
#fred=c/700E-9
#print('Synlig lys fra',fred*1E-12,' til',fblue*1E-12,' THz')

plt.figure('Planckkurver')
plt.plot(L*1E9,djdL1,label='3000 K')
plt.plot(L*1E9,djdL2,label='1500 K')
plt.legend(loc='best')
plt.title('Planckkurver',fontsize=20)
plt.xlabel('lambda (nm)',fontsize=24)
plt.ylabel('dj/dL (W/m^2 m)',fontsize=24)
#plt.savefig("planckkurver.eps")
plt.grid()
plt.show()