import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def XC_TS(t,X,a):
    Xa,Xi=X
    n,K,a1,a2,b1,b2=a
    dXa=a1*1/(np.power(K,n)+np.power(Xi,n))-b1*Xa
    dXi=a2*1/(np.power(K,n)+np.power(Xa,n))-b2*Xi
    return np.array([dXa,dXi])

t0=0
tm=10
n=2
K=0.5
a1=0.5
a2=0.5
b1=0.5
b2=0.5
X0=[1,0]
fname='../input/testdata.csv'

a=[n,K,a1,a2,b1,b2]
tspan=(t0,tm)
t=np.linspace(t0,tm)
sol=solve_ivp(XC_TS,tspan,X0,t_eval=t,args=(a,))

df=pd.DataFrame(np.append([sol.t],sol.y,axis=0).T,columns=['t','Xa','Xi'])
df.to_csv(fname,index=False)