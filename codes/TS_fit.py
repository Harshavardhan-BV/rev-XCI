#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#%%
def XC_DI(t,X,a):
    Xa,Xi=X
    n,K,a1,a2,b1,b2=a
    dXa=a1*1/(np.power(K,n)+np.power(Xi,n))-b1*Xa
    dXi=a2*1/(np.power(K,n)+np.power(Xa,n))-b2*Xi
    return np.array([dXa,dXi])

def XC_DA_DI(t,X,a):
    Xa,Xi=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXa=a1*np.power(Xa,n)/(np.power(K1,n)+np.power(Xa,n))+b1*1/(np.power(K2,n)+np.power(Xi,n))-c1*Xa
    dXi=a2*np.power(Xi,n)/(np.power(K3,n)+np.power(Xi,n))+b2*1/(np.power(K4,n)+np.power(Xa,n))-c2*Xi
    return np.array([dXa,dXi])

def XC_DA_DDI(t,X,a):
    Xa,Xi=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXa=a1*np.power(Xa,n)/(np.power(K1,n)+np.power(Xa,n))-b1*np.power(Xi,n)/(np.power(K2,n)+np.power(Xi,n))-c1*Xa
    dXi=a2*np.power(Xi,n)/(np.power(K3,n)+np.power(Xi,n))-b2*np.power(Xa,n)/(np.power(K4,n)+np.power(Xa,n))-c2*Xi
    return np.array([dXa,dXi])
#%%
def SSE(a,df,f):
    t=df['t'].values
    y_a= df.loc[:, df.columns != 't'].values
    y0=y_a[0]
    trang=(t[0],t[-1])
    sol=solve_ivp(f,trang,y0,t_eval=t,args=(a,))
    y_f=sol.y.T
    return np.sum(np.square(y_a-y_f))
#%%
df=pd.read_csv('../input/iPSC.csv')
#%%
n=2
K=0.5
a1=a2=1
b1=b2=1
a0=[n,K,a1,a2,b1,b2]
#%%
m=minimize(SSE,a0,args=(df,XC_DI,))
print(m.success)
print(m.fun)
print(m.x)
#%%
t=df['t'].values
y0= df.loc[:, df.columns != 't'].values[0]
trang=(t[0],t[-1])
sol=solve_ivp(XC_DI,trang,y0,t_eval=t,args=(m.x,))
plt.plot(t,sol.y[:,0],label='Xa')
plt.plot(t,sol.y[:,1],label='Xi')
# %%
