# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:10:45 2020

@author: DUC TAM VU
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:24:15 2020

@author: DUC TAM VU
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


#données caractéristiques

df = 0.9
Lf=48
Ef=210000
Em=28000
vf=0.005
tau0=4
beta=0.001
nu=0.25
I=np.pi*df**4/64

#variable frottement

eta=Ef/Em*vf/(1-vf)
delta0=(1+eta)*tau0*Lf**2/(2*df*Ef)
nd=1/2*np.pi*df*np.sqrt(Ef*tau0*df*(1+eta)/2)

def Nd(d) :         #fonction de décollement
    N=nd*np.sqrt(d)
    return N
def Ne(d):          #fonction de frottement
    N=nd*np.sqrt(delta0)*(1-4*(d-delta0)/Lf)
    return N

#configure les paramètres dépendent à N(x)
   
km=Em*0.2
zeta=np.exp(np.log(km/(4*Ef*I))/4)

def mm(n) :
    m=np.sqrt(n/Ef/I)
    return m

def long(phi,gamma,w,s):        #longueur de partie extraite : problème de converge
    l=1/2*(df*np.tan(phi)+w*np.cos(gamma)+s*np.sin(gamma))
    return l

def k1(N,l,m):
    K1=-1/(4*zeta**2*N**2/km*(m*np.cosh(m*l)+2*zeta*np.sinh(m*l))+2*N*m*np.cosh(m*l))
    return K1

def k2(K,l,N,m):
    K2=-4*zeta/km*K*N*(m*np.cosh(m*l)+zeta*np.sinh(m*l))+2*K*np.sinh(m*l)+l/N
    return K2

def dt(delta,alpha):            #biais partie ancrage/section symétrique
    dt=delta*np.sin(alpha)
    return dt

#fonction de force axiale
def F(w,s,phi): 
    gamma = np.arctan(s/w)          #angle de chargement
    alpha = np.absolute(phi-gamma)       #angle effective
  #  if alpha.any() <0 : alpha = np.absolute(alpha)
 #   else : pass
    DT=np.sqrt(w**2+s**2)/2
    if DT.any()<= delta0:                
        N=Nd(DT)
    else:
        N=Ne(DT)
        l=long(phi,gamma,w,s)
        m=mm(N)
        K1=k1(N,l,m)
        K2=k2(K1,l,N,m)
        P=dt(DT,alpha)/K2
        f=(N*np.cos(alpha)+P*np.sin(alpha))/2
    return f

#plot section

fig=plt.figure(figsize=(12,9))
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
#ax=fig.add_subplot(1,1,1, projection= '3d')
plt.title("Force axial de fibre sous le déplacement tangentiel des lèvres de la fissure à CMOD = 0.1", fontsize=14, y=-0.1)
ax=fig.gca()
ax.set_xlabel('g (mm)', fontsize=12)
ax.xaxis.labelpad = 5
#ax.invert_xaxis()
ax.set_ylabel('F (N)',fontsize=12)
ax.yaxis.labelpad = 5
#ax.set_zlabel('Force axial F(N)',fontsize=12)
#ax.zaxis.labelpad = 10

#résultat
angle=[-np.pi/6,-np.pi/12,0,np.pi/12, np.pi/6]    #angle inclinaison initial
w=0.5
s=np.linspace(0.001, 5, 1001)
for phi in angle :
    plt.plot(s,F(w,s,phi)*np.sin(np.arctan(s/w)), linewidth=2, label='\u03C6='+str(phi))
    plt.legend(loc="lower right", prop = {"size":12})
            























