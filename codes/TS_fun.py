import numpy as np
# Defining Differential equations
def A(X,K,n):
    return np.power(X,n)/(np.power(K,n)+np.power(X,n))

def I(X,K,n):
    return np.power(K,n)/(np.power(K,n)+np.power(X,n))

def N(X,K,n):
    return 0

def XC(t,X,a,f):
    Xi,Xa=X
    funs={'A': A, 'I': I, 'N': N}
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXi=a1*funs[f[0]](Xa,K1,n)+b1*funs[f[2]](Xi,K3,n)-c1*Xi
    dXa=a2*funs[f[1]](Xi,K2,n)+b2*funs[f[3]](Xa,K4,n)-c2*Xa
    return np.array([dXi,dXa])
