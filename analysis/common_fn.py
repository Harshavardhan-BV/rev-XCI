# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=22

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

# Plot Timeseries
def timeseries(fns,inp,txt):
    n=len(fns)
    blu = plt.cm.Blues(np.linspace(0.2, 1, n))
    orang = plt.cm.Oranges(np.linspace(0.2, 1, n))
    iname='../input/'+inp+'.csv'
    idf=pd.read_csv(iname)
    figname='../figures/'+txt+'-'+inp+'-timeseries.svg'
    # Solver input parameters
    t=idf['t'].values
    trang=(t[0],t[-1])
    teval=np.linspace(t[0],t[-1],100)
    y_a= idf.loc[:,['Xi','Xa']].values
    y0=y_a[0]
    # Plot Actual datapoints
    fig=plt.figure(figsize=(15,10))
    plt.scatter(t,y_a[:,0],label='Xi',c='tab:blue')
    plt.scatter(t,y_a[:,1],label='Xa',c='tab:orange')
    for i in range(n):
        f=fns[i]
        oname='../output/'+inp+'-'+f+'-parm.csv'
        odf=pd.read_csv(oname)
        a=odf.values[0]
        sol=solve_ivp(XC,trang,y0,t_eval=teval,args=(a,f))
        plt.plot(sol.t,sol.y[0],c=blu[i],label='Xi-'+f)
        plt.plot(sol.t,sol.y[1],c=orang[i],label='Xa-'+f)
    # Labels
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('Time (days)')
    plt.ylabel('X:A')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)

# Returns the sum of square error and R-square of the fit
def SSE(f,inp):
    iname='../input/'+inp+'.csv'
    idf=pd.read_csv(iname)
    oname='../output/'+inp+'-'+f+'-parm.csv'
    odf=pd.read_csv(oname)
    # Solver input parameters
    t=idf['t'].values
    trang=(t[0],t[-1])
    y_a= idf.loc[:,['Xi','Xa']].values
    y0=y_a[0]
    a=odf.values[0]
    # Solve differetial equation 
    sol=solve_ivp(XC,trang,y0,t_eval=t,args=(a,f))
    # Get 'fit' equation
    y_f=sol.y.T
    # Get sum of square error
    sse=np.sum(np.square(y_a-y_f))
    # R-Square?
    rsq=r2_score(y_a,y_f)
    return [sse,rsq]

def hmap(fns,axs,lbl,inp,txt):
    sses=np.empty_like(fns,dtype=float)
    rsqrs=np.empty_like(fns,dtype=float)
    for i in range(fns.shape[0]):
        for j in range(fns.shape[1]):
            sses[i,j],rsqrs[i,j]=SSE(fns[i,j],inp)
    fig=plt.figure()
    s=sns.heatmap(sses,xticklabels=axs,yticklabels=axs,cmap='coolwarm_r',annot=True)
    s.set(xlabel=lbl[0],ylabel=lbl[1])
    plt.tight_layout()
    figname='../figures/'+txt+'-'+inp+'-sse-hmap.svg'
    plt.savefig(figname)
    plt.close(fig)
    fig=plt.figure()
    s=sns.heatmap(rsqrs,xticklabels=axs,yticklabels=axs,cmap='coolwarm',annot=True)
    s.set(xlabel=lbl[0],ylabel=lbl[1])
    plt.tight_layout()
    figname='../figures/'+txt+'-'+inp+'-rsq-hmap.svg'
    plt.savefig(figname)
    plt.close(fig)