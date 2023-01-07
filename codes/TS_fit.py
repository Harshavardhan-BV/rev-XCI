#!/usr/bin/env python
#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
#%% # Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', help='Function to use for fit')
parser.add_argument('-i', help='Input file')
# parser.add_argument('-n', help='Number of parameter sets',type=int)
parser.add_argument('-t', help='Number of process threads',type=int)
args = parser.parse_args()
fname=args.i
thr=args.t
func=args.f
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
#%% Objective function
def SSE(a,df,f):
    # Timepoints from actual data
    t=df['t'].values
    # Actual y values
    y_a= df.loc[:,['Xi','Xa']].values
    # Initial condition
    y0=y_a[0]
    # Time range to solve for (range of given data)
    trang=(t[0],t[-1])
    # Solve differetial equation 
    sol=solve_ivp(f,trang,y0,t_eval=t,args=(a,))
    # Get 'fit' equation
    y_f=sol.y.T
    # Get sum of square error
    if y_f.shape==y_a.shape:
        return np.sum(np.square(y_a-y_f))
    # Return nan if solver error
    else:
        return np.nan
#%% Read input data
funcs = {'XC_DI': XC_DI, 'XC_DA_DI': XC_DA_DI, 'XC_DA_DDI': XC_DA_DDI, 'XC_DI_DI': XC_DI_DI}
f=funcs[func]
df=pd.read_csv('../input/'+fname+'.csv')
filename=fname+'-'+f.__name__
if f.__name__=='XC_DI':
    asize=7
    cname=np.array(['n','K1','K2','a1','a2','c1','c2'],dtype=str)
else:
    asize=11
    cname=np.array(['n','K1','K2','K3','K4','a1','a2','b1','b2','c1','c2'],dtype=str)
filename='../output/'+filename+'-parm.csv'
# %%
bounds=np.full([asize,2],[0,10])
res=differential_evolution(SSE,bounds,args=(df,f),init='sobol',workers=thr,updating='deferred')
# %%
df=pd.DataFrame([res.x],columns=cname)
df.to_csv(filename,index=False)
# %%
