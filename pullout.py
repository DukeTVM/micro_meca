# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:24:15 2020

@author: DUC TAM VU
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as scp


#données caractéristiques

df = 0.75
Lf=40
Ef=210000
Em=28000
vf=0.005
tau0=4
beta=0.001
nu=0.25

#variable frottement

eta=Ef/Em*vf/(1-vf)
delta0=(1+eta)*tau0*Lf**2/(2*df*Ef)
nd=1/2*np.pi*df*np.sqrt(Ef*tau0*df*(1+eta)/2)

def Nd(d) :
    N=nd*np.sqrt(d)
    return N
def Ne(d):
    N=nd*np.sqrt(delta0)*(1-4*(d-delta0)/Lf)
    return N

#variable tangent

I=np.pi*df**4/64
def mm(n) :
    m=np.sqrt(n/Ef/I)
    return m
Im=2.182
km=Em/5
zeta=np.exp(np.log(km/(4*Ef*I))/4)
def long(phi,gamma,w,s):
    l=1/2*(df*np.tan(phi)+w*np.cos(gamma)+s*np.sin(gamma))
    return l
def k1(N,l,m):
    K1=-1/(4*zeta**2*N**2/km*(m*np.cosh(m*l)+zeta*np.sinh(m*l))+2*N*m*np.cosh(m*l))
    #K1=(2*zeta**2*N-km)/(2*km*N*m*np.cosh(m*l)+8*zeta**3*N*2*np.sinh(m*l))
    return K1
def k2(K,l,N,m):
    K2=-4*zeta/km*K*N*(m*np.cosh(m*l)+zeta*np.sinh(m*l))+2*K*np.sinh(m*l)+l/N
 #   K2=2*zeta/km+l/N-(4*zeta**2/km*N-2)*K*np.sinh(m*l)
    return K2
def dt(delta,alpha):
#(gam,w,s):
    dt=delta*np.sin(alpha)#+s*np.cos(gam)
    return dt
#result
phi=-np.pi/12 # np.pi/15, np.pi/12, np.pi/6,np.pi/4)
fig=plt.figure()
W=[0.05,0.1,0.2,0.5]
#w=0.1
for w in W:
# phi in PH:
    s=np.linspace(0,2,100)
    Np=[0]
    P=[0]
    F=[0]
    DT=[0]
    for i in range(1,len(s)):
            gamma = np.arctan(s[i]/w)
            alpha=phi+gamma
            if alpha <0 : alpha = np.absolute(alpha)
            else : pass
            DT.append(np.sqrt(w**2+s[i]**2)/2)
            if DT[i] <= delta0:                
                N=Nd(DT[i])
            else:
                N=Ne(DT[i])
                Np.append(N)
                l=long(phi,gamma,w,s[i])
                m=mm(N)
                K1=k1(N,l,m)
                K2=k2(K1,l,N,m)
                P.append(dt(DT[i],alpha)/K2)#s[i]))
                f=N*np.cos(alpha)+P[i]*np.sin(alpha)
                F.append(f)
    plt.plot(DT,F)
            
























