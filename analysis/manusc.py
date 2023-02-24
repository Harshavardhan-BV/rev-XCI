# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=40
# plt.rcParams["text.usetex"] = True

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

def hmap(fns,axs,lbl,inp,txt,titl=''):
    siz=4.5*len(axs)
    sses=np.empty_like(fns,dtype=float)
    rsqrs=np.empty_like(fns,dtype=float)
    for i in range(fns.shape[0]):
        for j in range(fns.shape[1]):
            sses[i,j],rsqrs[i,j]=SSE(fns[i,j],inp)
    fig=plt.figure(figsize=(siz+2,siz))
    s=sns.heatmap(sses,xticklabels=axs,yticklabels=axs,cmap='coolwarm_r',annot=True)
    s.set(xlabel=lbl[0],ylabel=lbl[1])
    plt.title(titl)
    plt.tight_layout()
    figname='../writing/draft/figures/'+txt+'-'+inp+'-sse-hmap.pdf'
    plt.savefig(figname)
    plt.close(fig)
    fig=plt.figure(figsize=(siz+2,siz))
    s=sns.heatmap(rsqrs,xticklabels=axs,yticklabels=axs,cmap='coolwarm',annot=True,vmin=0.5,vmax=0.95)
    s.set(xlabel=lbl[0],ylabel=lbl[1],)
    plt.title(titl)
    plt.tight_layout()
    figname='../writing/draft/figures/'+txt+'-'+inp+'-rsq-hmap.pdf'
    plt.savefig(figname)
    plt.close(fig)

def timeseries_topo(f,inp,titl=''):
    iname='../input/'+inp+'.csv'
    idf=pd.read_csv(iname)
    figname='../writing/draft/figures/'+f+'-'+inp+'-timeseries.pdf'
    # Solver input parameters
    t=idf['t'].values
    trang=(t[0],t[-1])
    teval=np.linspace(t[0],t[-1],100)
    y_a= idf.loc[:,['Xi','Xa']].values
    y0=y_a[0]
    # Plot Actual datapoints
    fig=plt.figure(figsize=(15,10))
    plt.scatter(t,y_a[:,0],label='$X_i$',c='tab:blue',s=100)
    plt.scatter(t,y_a[:,1],label='$X_a$',c='tab:orange',s=100)
    oname='../output/'+inp+'-'+f+'-parm.csv'
    odf=pd.read_csv(oname)
    a=odf.values[0]
    sol=solve_ivp(XC,trang,y0,t_eval=teval,args=(a,f))
    plt.plot(sol.t,sol.y[0],c='tab:blue',label='$X_i$-fit',lw=5)
    plt.plot(sol.t,sol.y[1],c='tab:orange',label='$X_a$-fit',lw=5)
    # Labels
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('X:A')
    plt.title(titl)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)

def parmcomp(f,ts=False):
    if ts:
        ts='_timeshifted'
    else:
        ts=''
    figname='../writing/draft/figures/'+f+ts+'-parmcomp.pdf'
    iname='../output/iPSC'+ts+'-'+f+'-parm.csv'
    idf=pd.read_csv(iname)
    idf['Data']='Full'
    pname='../output/Partial'+ts+'-'+f+'-parm.csv'
    pdf=pd.read_csv(pname)
    pdf['Data']='Partial'
    fig=plt.figure(figsize=(15,10))
    df=pd.concat([idf,pdf])
    df=pd.melt(df,'Data')
    sns.barplot(df,x='variable',y='value',hue='Data')
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)


def timeseries_noise_violin(f,inp):
    fname='../output/'+inp+'-'+f+'-noise.csv'
    figname='../writing/draft/figures/'+f+'-'+inp+'-noise-violin.pdf'
    df=pd.read_csv(fname)
    df = df.astype({"t": int})
    fig, ax = plt.subplots(1,2,figsize=(30,10),sharey=True)
    l1 = sns.violinplot(df,x='t',y='Xi',color='tab:blue',scale='width',ax=ax[0])
    l2 = sns.violinplot(df,x='t',y='Xa',color='tab:orange',scale='width',ax=ax[1])
    ax[0].set_ylabel('X:A')
    ax[0].set_xlabel('Time (days)')
    ax[1].set_ylabel('')
    ax[1].set_xlabel('Time (days)')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)

timeseries_topo('IIII', 'iPSC_timeshifted','Full: Cross-Inhibition w/ Self-Inhibition')
timeseries_topo('IIAA', 'iPSC_timeshifted','Full: Cross-Inhibition w/ Self-Activation')
timeseries_topo('IINN', 'iPSC_timeshifted','Full: Cross-Inhibition')
timeseries_topo('IIII', 'Partial_timeshifted', 'Partial: Cross-Inhibition w/ Self-Inhibition')

timeseries_noise_violin('IIII', 'iPSC_timeshifted')

axs=['A','I']
lbl=['Incoming connection to $X_a$','Incoming connection to $X_i$']
combi2d=np.array([['AAAA','AIAA'],['IAAA','IIAA']])
hmap(combi2d,axs,lbl,'iPSC_timeshifted','vary_cross-AA',titl='Full: Self-Activation')
hmap(combi2d,axs,lbl,'Partial_timeshifted','vary_cross-AA',titl='Partial: Self-Activation')

combi2d=np.array([['AAII','AIII'],['IAII','IIII']])
hmap(combi2d,axs,lbl,'iPSC_timeshifted','vary_cross-II',titl='Full: Self-Inhibition')
hmap(combi2d,axs,lbl,'Partial_timeshifted','vary_cross-II',titl='Partial: Self-Inhibition')

lbl=['Self-connection of $X_a$','Self-connection of $X_i$']
combi2d=np.array([['IIAA','IIAI','IIAN'],['IIIA','IIII','IIIN'],['IINA','IINI','IINN']])
axs=['A','I','N']
hmap(combi2d,axs,lbl,'iPSC_timeshifted','vary_self',titl='Full: Cross-Inhibition')
hmap(combi2d,axs,lbl,'Partial_timeshifted','vary_self',titl='Partial: Cross-Inhibition')
parmcomp('IIII',ts=True)

from itertools import product
combi=np.array(tuple(product(axs,repeat=4)))
combi=np.apply_along_axis(''.join,1,combi)
combi2d=np.reshape(combi,(9,9))
axs=np.array(tuple(product(axs,repeat=2)))
axs=np.apply_along_axis(''.join,1,axs)
lbl=['Self','Cross']
hmap(combi2d,axs,lbl,'iPSC_timeshifted','vary_all')
hmap(combi2d,axs,lbl,'Partial_timeshifted','vary_all')