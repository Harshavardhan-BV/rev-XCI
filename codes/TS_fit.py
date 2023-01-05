#!/usr/bin/env python
#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize,fmin
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool
plt.rcParams["svg.hashsalt"]=''
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', help='Function to use for fit')
parser.add_argument('-i', help='Input file')
parser.add_argument('-n', help='Number of parameter sets',type=int)
parser.add_argument('-t', help='Number of process threads',type=int)
args = parser.parse_args()
#%% Defining Differential equations
# Double cross-inhibition
def XC_DI(t,X,a):
    Xi,Xa=X
    n,K1,K2,a1,a2,c1,c2=a
    dXi=a1*1/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=a2*1/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
# Double self-activation + cross-inhibition
def XC_DA_DI(t,X,a):
    Xi,Xa=X
    n,K1,K2,K3,K4,a1,a2,b1,b2,c1,c2=a
    dXi=b1*np.power(Xi,n)/(np.power(K3,n)+np.power(Xi,n))+a1*1/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=b2*np.power(Xa,n)/(np.power(K4,n)+np.power(Xa,n))+a2*1/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
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
    dXi=b1*1/(np.power(K3,n)+np.power(Xi,n))+a1*1/(np.power(K1,n)+np.power(Xa,n))-c1*Xi
    dXa=b2*1/(np.power(K4,n)+np.power(Xa,n))+a2*1/(np.power(K2,n)+np.power(Xi,n))-c2*Xa
    return np.array([dXi,dXa])
#%% Objective function
def SSE(a,df,f):
    # Timepoints from actual data
    t=df['t'].values
    # Actual y values
    y_a= df.loc[:, df.columns != 't'].values
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
#%% Solve and Display 
def plot_fit(df,f,a,fname,savefig=True,savedat=True):
    t=df['t'].values
    y_a= df.loc[:, df.columns != 't'].values
    y0=y_a[0]
    trang=(t[0],t[-1])
    tlin=np.linspace(t[0],t[-1])
    # Solve with given parameters
    sol=solve_ivp(f,trang,y0,t_eval=tlin,args=(a,))
    # Plot Actual datapoints
    plt.scatter(t,y_a[:,0],label='Xi')
    plt.scatter(t,y_a[:,1],label='Xa')
    # Plot Fit data
    plt.plot(sol.t,sol.y[0],label='Xi-fit')
    plt.plot(sol.t,sol.y[1],label='Xa-fit')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('X:A')
    # Convert to dataframe
    f_df=pd.DataFrame(np.append([sol.t],sol.y,axis=0).T,columns=['t','Xi','Xa'])
    # Save outputs
    if savefig:
        plt.savefig('../output/'+fname+'.svg')
    if savedat:
        f_df.to_csv('../output/'+fname+'-fit.csv')
    return f_df

#%% Read input data
fname=args.i
funcs = {'XC_DI': XC_DI, 'XC_DA_DI': XC_DA_DI, 'XC_DA_DDI': XC_DA_DDI, 'XC_DI_DI': XC_DI_DI}
f=funcs[args.f]
df=pd.read_csv('../input/'+fname+'.csv')
fname=fname+'-'+f.__name__
if f.__name__=='XC_DI':
    asize=7
    cname=np.array([['n','K1','K2','a1','a2','c1','c2']],dtype=str)
else:
    asize=11
    cname=np.array([['n','K1','K2','K3','K4','a1','a2','b1','b2','c1','c2']],dtype=str)
filename='../output/'+fname+'-parm.csv'
np.savetxt(filename,cname,fmt='%s',delimiter=',')
#%% Initial guess 
a0s=np.random.randint(0,100,(args.n,asize))
# %%
def min_fit(a0):
    m=fmin(SSE,a0,args=(df,f,))
    with open(filename, 'a') as fil:
        np.savetxt(fil,[m],delimiter=',')
    return m
# %%
if __name__ == '__main__':
    pool = Pool(args.t)
    m_arr=pool.map(min_fit,a0s) 
    pool.close()
    pool.join()
# %%
#m_arr=pd.DataFrame(m_arr,columns=cname)
#m_arr.to_csv('../output/'+fname+'-parm.csv',index=False)
