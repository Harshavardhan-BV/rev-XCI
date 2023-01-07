# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=22
#%% Defining Differential equations
# Double cross-inhibition
def XC_DI(t,X,a):
    Xi,Xa=X
    n,K1,K2,a1,a2,c1,c2=a
    dXi=a1*np.power(K1,n)/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=a2*np.power(K2,n)/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
# Double self-activation + cross-inhibition
def XC_DA_DI(t,X,a):
    Xi,Xa=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXi=b1*np.power(Xi,n)/(np.power(K3,n)+np.power(Xi,n))+a1*np.power(K1,n)/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=b2*np.power(Xa,n)/(np.power(K4,n)+np.power(Xa,n))+a2*np.power(K2,n)/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
# Double self-activation + (-ve) cross-inhibition
def XC_DA_DDI(t,X,a):
    Xi,Xa=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXi=b1*np.power(Xi,n)/(np.power(K3,n)+np.power(Xi,n))-a1*np.power(Xa,n)/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=b2*np.power(Xa,n)/(np.power(K4,n)+np.power(Xa,n))-a2*np.power(Xi,n)/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
# Double self-inhibition + cross-inhibition
def XC_DI_DI(t,X,a):
    Xi,Xa=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXi=b1*np.power(K3,n)/(np.power(K3,n)+np.power(Xi,n))+a1*np.power(K1,n)/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=b2*np.power(K4,n)/(np.power(K4,n)+np.power(Xa,n))+a2*np.power(K2,n)/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
# %%
func=XC_DI
inp='iPSC_timeshifted'
iname='../input/'+inp+'.csv'
oname='../output/'+inp+'-'+func.__name__+'-parm.csv'
idf=pd.read_csv(iname)
odf=pd.read_csv(oname)
figname='../figures/'+inp+'-'+func.__name__+'-timeseries.svg'
#%%
t=idf['t'].values
trang=(t[0],t[-1])
teval=np.linspace(t[0],t[-1])
y_a= idf.loc[:, idf.columns != 't'].values
y0=y_a[0]
#%%
a=odf.values[0]
#%%
plt.figure(figsize=(15,10))
sol=solve_ivp(func,trang,y0,t_eval=teval,args=(a,))
plt.plot(sol.t,sol.y[0],c='tab:blue',label='Xi-fit')
plt.plot(sol.t,sol.y[1],c='tab:orange',label='Xa-fit')
# Plot Actual datapoints
plt.scatter(t,y_a[:,0],label='Xi',c='tab:blue')
plt.scatter(t,y_a[:,1],label='Xa',c='tab:orange')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('X:A')
plt.savefig(figname)
# %%
