import numpy as np
# Defining Differential equations
def A(X,K,n):
    return np.power(X,n)/(np.power(K,n)+np.power(X,n))

def I(X,K,n):
    return np.power(K,n)/(np.power(K,n)+np.power(X,n))

def N(X,K,n):
    return 1

def XC(t,X,a,f):
    Xi,Xa=X
    funs={'A': A, 'I': I, 'N': N}
    n,K1,K2,K3,K4,g1,g2,k1,k2=a
    dXi=g1*funs[f[0]](Xa,K1,n)*funs[f[2]](Xi,K3,n)-k1*Xi
    dXa=g2*funs[f[1]](Xi,K2,n)*funs[f[3]](Xa,K4,n)-k2*Xa
    return np.array([dXi,dXa])

def XC_noise(t,X,a,f,mu=0,sig=0.1):
    dX = XC(t,X,a,f)
    return dX + np.random.normal(mu,sig,dX.shape)