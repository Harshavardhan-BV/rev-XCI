# %%
import numpy as np
import pandas as pd
from TS_fun import XC_noise
from itertools import repeat
from multiprocessing import Pool
from scipy.integrate import solve_ivp
# %%
def noise_solve(y0,t,fname,f):
    parms_file='../output/'+fname+'-'+f+'-parm.csv'
    parms=pd.read_csv(parms_file)
    a=parms.values[0]
    trang=(t[0],t[-1])
    sol=solve_ivp(XC_noise,trang,y0,t_eval=t,args=(a,f))
    return sol.y.T
# %%%
def noise_solver_multi(fname,f,n):
    iname = '../input/'+fname+'.csv'
    df = pd.read_csv(iname)
    t=df.t.values
    y_0= df.loc[0,['Xi','Xa']].values
    with Pool() as p:
        sol = p.starmap(noise_solve,zip(repeat(y_0,n), repeat(t), repeat(fname),repeat(f)))
    solarr=np.array(sol).reshape(-1,2)
    solt=np.repeat(t.reshape(1,-1),n,axis=0).reshape(-1,1)
    df = pd.DataFrame(np.hstack((solt,solarr)),columns=['t','Xi','Xa'])
    oname = '../output/'+fname+'-'+f+'-noise.csv'
    df.to_csv(oname,index=False)
# %%
f = 'IIII'
fname='iPSC_timeshifted'
n=1000

noise_solver_multi(fname, f, n)