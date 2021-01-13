# -*- coding: utf-8 -*-
"""
Duc-Tam VU - micromechanical modeling pullout behavior
Derivation of the pullout process of single fiber under combined displacement of crack: opening and sliding
Plot 2D curve of the analytical model for hooked-end and straight fiber 


"""

import numpy as np
from scipy import integrate 
import matplotlib.pyplot as plt



#donnés caractéristiques
df=0.9              #diametre
Lf=60               #longueur totale
L1=2                #longueur du crochet 1
L2=2                #longueur du crochet 2
psi=np.pi/4         #angle du crochet
Ef=210000           #module d'élasticité de l'acier
Em=30000            #module d'élasticité de la matrice
vf=0.005            #teneur en fibre
tau0=7              #contrainte initial à l'interface
mu=0.15             #coefficient Coulomb
sigp=1150           #limite plastique de l'acier
I=np.pi*df**4/64    #moment d'inertie

#variable de l'orientation
kg=8               #coefficient de la dispersion de la loi proba
phi0=np.pi/15              #orientation préférée

#variable de l'extraction
eta = Ef/Em*vf/(1-vf)
k0 = np.pi*df*np.sqrt((1+eta)*tau0*df*Ef/2)

km=Em*0.25          #rigidité de la matrice cimentaire
zeta = np.exp(np.log(km/(4*Ef*I))/4)    #constant de l'équation différentielle

###########################################################
#les fonctions du composant normal
def Nd(d) :         #fonction de décollement
    return k0*np.sqrt(d)

Q=np.pi*sigp*df**2/(24*np.sin(psi/2)*(1-mu*np.sin(psi/2)))
R=2/(1-mu*np.sin(psi/2))
mult=L1*L2*(L1+L2)

def Nr(d,L):        #fonction de redressement pour les fibres aux crochets
    delta0=L**2*(1+eta)*2*tau0/(Ef*df)   
    Nr=Q*(L1-R*(L1+L2))/mult*(d-delta0)**2+Q*(R*(L1+L2)**2-L1**2)/mult*(d-delta0)+Nd(delta0)
    return Nr

def NeD(d,L) :         #fonction de frottement
    delta0=L**2*(1+eta)*2*tau0/(Ef*df) 
    return np.pi*df*(1+eta)*tau0*(L-d+delta0)

def NeH(d,L) :         #fonction de frottement
    delta0=L**2*(1+eta)*2*tau0/(Ef*df) 
    return np.pi*df*(1+eta)*tau0*(L-d+delta0+L1+L2)*(1+Q/k0/np.sqrt(delta0))

def z0(d):                      #position de la fibre en décollement
    return Lf/2-np.sqrt(Ef*df*d/tau0/2*(1+eta))

def z1(d):                      #position de la fibre en redressement
    delt=Lf**2*(1+eta)*tau0/(Ef*df)/2
    if d < L1+L2 : 
         z=0
    else:z=Lf/2-np.sqrt((d-L1-L2)*Ef*df/((tau0+tau0*Q/k0(vf)/np.sqrt(delt))*2*(1+eta)))
    return z

def alpha(phi,theta,gamma):         #angle effective
    return np.arccos(np.cos(phi)*np.cos(gamma)+np.sin(phi)*np.sin(gamma)*np.cos(theta))

###########################################################
#loi de probabilité
def proba(phi, theta):        
    p=kg/(2*np.pi*np.sinh(kg))*np.cosh(kg*(np.cos(phi)*np.cos(phi0)-np.sin(phi)*np.sin(theta)*np.sin(phi0)))
    return p

###########################################################
def K(N,d):           #coefficient K de l'effort tranchant
    m=np.sqrt(N/Ef/I)
    l=d/2+df/2
    K=l/((2*zeta*N*(m*np.cosh(m*l)+zeta*np.sinh(m*l))-km*np.sinh(m*l))/\
    (2*zeta**2*N*(m*np.cosh(m*l)+2*zeta*np.sinh(m*l))+km*m*np.cosh(m*l))+l)
    return K

###########################################################
#les composants de l'intégration finale
VF=vf/(np.pi*df**2/4)*2/Lf
def coh1D(d,gamma):            #contrainte au décollement
    f=lambda L,phi,theta : (np.sin(alpha(phi,theta,gamma))**2*K(Nd(d),d)+np.cos(alpha(phi,theta,gamma)))*Nd(d)*proba(phi,theta)*np.sin(phi)
    fd=integrate.nquad(f,[[Lf/2-z0(d),Lf/2],[0,np.pi/2],[0,2*np.pi]])
    return fd[0]*VF

def coh1E(d,gamma):            #contrainte au glissement en 1eme integration - fibre droite
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(NeD(d,L),d))*NeD(d,L)*proba(phi,theta)*np.sin(phi)
    fe=integrate.nquad(f,[[d,(Lf/2-z0(d))],[0,np.pi/2],[0,2*np.pi]])
    return fe[0]*VF

def coh2E(d,gamma):            #contrainte au glissement en 2eme integration - fibre droite
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(NeD(d,L),d))*NeD(d,L)*proba(phi,theta)*np.sin(phi)
    fe=integrate.nquad(f,[[d,Lf/2],[0,np.pi/2],[0,2*np.pi]])
    return fe[0]*VF

def coh1R(d,gamma):             #contrainte au redressement en 1eme integration
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(Nr(d,L),d))*Nr(d,L)*proba(phi,theta)*np.sin(phi)
    fr=integrate.nquad(f,[[d,(Lf/2-z0(d))],[0,np.pi/2],[0,2*np.pi]])
    return fr[0]*VF

def coh2R(d,gamma):             #contrainte au redressement en 2eme intégration
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(Nr(d,L),d))*Nr(d,L)*proba(phi,theta)*np.sin(phi)
    fr=integrate.nquad(f,[[d,Lf/2],[0,np.pi/2],[0,2*np.pi]])
    return fr[0]*VF

def coh3R(d,gamma):             #contrainte au redressement en 3eme intégration
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(Nr(d,L),d))*Nr(d,L)*proba(phi,theta)*np.sin(phi)
    fr=integrate.nquad(f,[[(Lf/2-z1(d)),Lf/2],[0,np.pi/2],[0,2*np.pi]])
    return fr[0]*VF

def coh3E(d,gamma):            #contrainte au glissement en 3eme integration - fibre au crochet
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(NeH(d,L),d))*NeH(d,L)*proba(phi,theta)*np.sin(phi)
    fe=integrate.nquad(f,[[d,(Lf/2-z1(d))],[0,np.pi/2],[0,2*np.pi]])
    return fe[0]*VF

def coh4E(d,gamma):            #contrainte au glissement en 4eme integration - fibre au crochet
    f = lambda L,phi,theta: (np.cos(alpha(phi,theta,gamma))+np.sin(alpha(phi,theta,gamma))**2*K(NeH(d,L),d))*NeH(d,L)*proba(phi,theta)*np.sin(phi)
    fe=integrate.nquad(f,[[d,Lf/2],[0,np.pi/2],[0,2*np.pi]])
    return fe[0]*VF

######################################################################################
deltaE = Lf**2*(1+eta)*tau0/(2*Ef*df) #déplacement où tout les fibres sont décollées

def sigmaH(d,gamma):            #fonction finale de la contrainte - fibre crochet
    if d<=deltaE:
        sigma=coh1D(d,gamma)+coh1R(d,gamma)
    elif deltaE<d<=L1+L2:   
        sigma=coh2R(d,gamma)
    elif L1+L2<d<=deltaE+L1+L2: #déplacement où tout les fibres sont redressées
        sigma=coh3R(d,gamma)+coh3E(d,gamma)
    else:
        sigma=coh4E(d,gamma)
    return sigma

def sigmaD(d,gamma):        #fonction finale de la contrainte - fibre droite
    if d<=deltaE:
        sigma=coh1D(d,gamma)+coh1E(d,gamma)
    else:
        sigma=coh2E(d,gamma)
    return sigma
    

#######################################################################################
#boucle de calcul
lim=51
w=np.linspace(0,5,lim)
s=np.linspace(0,5,lim)
stressD=[]
stressH=[]
for i in range(lim):
    if w[i]==0:
        stressD.append(0)   
        stressH.append(0)
    else:
        gamma=np.arctan(s[i]/w[i])
        d=np.sqrt(w[i]**2+s[i]**2)  
        stressD.append(sigmaD(d,gamma)*np.sin(gamma)) 
        stressH.append(sigmaH(d,gamma)*np.sin(gamma))
#plot des courbes
fig=plt.figure(figsize=(12,8))
plt.xlabel('Sliding s(mm)', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('Shear stress (MPa)',fontsize=20)
plt.yticks(fontsize=20)
plt.grid(linestyle='--', linewidth=1.5)
plt.plot(s,stressD,linewidth=3,color='blue',marker='o',markersize=8,label='Straight')
plt.plot(s,stressH,linewidth=3,color='red',marker='x',markersize=8,label='Hooked-end')
plt.legend(loc="upper right", prop = {"size":22}) 


