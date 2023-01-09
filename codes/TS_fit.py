#!/usr/bin/env python
#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from TS_fun import XC
thr=16
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
    sol=solve_ivp(XC,trang,y0,t_eval=t,args=(a,f))
    # Get 'fit' equation
    y_f=sol.y.T
    # Get sum of square error
    if y_f.shape==y_a.shape:
        return np.sum(np.square(y_a-y_f))
    # Return nan if solver error
    else:
        return np.nan
#%% Main function for minimizing 
def de_fit(f,fname):
    #%% Read input data
    df=pd.read_csv('../input/'+fname+'.csv')
    filename=fname+'-'+f
    cname=np.array(['n','K1','K2','K3','K4','a1','a2','b1','b2','c1','c2'],dtype=str)
    filename='../output/'+filename+'-parm.csv'
    # %%
    bounds=np.full([11,2],[0,10])
    res=differential_evolution(SSE,bounds,args=(df,f),init='sobol',workers=thr,updating='deferred')
    # %%
    df=pd.DataFrame([res.x],columns=cname)
    df.to_csv(filename,index=False)
# %%
