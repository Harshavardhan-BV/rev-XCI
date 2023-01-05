# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=22
# %%
func='XC_DI_DI'
# ts='_timeshifted'
ts=''
fname='../output/iPSC'+ts+'-'+func+'-parm.csv'
idf=pd.read_csv(fname)
fname='../output/Partial'+ts+'-'+func+'-parm.csv'
pdf=pd.read_csv(fname)
# %%
idf1=idf.melt()
idf1['cat']='iPSC'
pdf1=pdf.melt()
pdf1['cat']='Partial'
df=pd.concat([idf1,pdf1],ignore_index=True)
# %%
fig=plt.figure(figsize=(20,10))
sns.boxplot(data=df,x='variable',y='value',hue='cat',showfliers = False)
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'-parms.svg'
plt.savefig(figname)
# %%
idf1=idf.copy()
pdf1=pdf.copy()
idf1=idf1.loc[:,'a1':'c2'].div(idf1.loc[:,'a1':'c2'].max(axis=1),axis=0)
pdf1=pdf1.loc[:,'a1':'c2'].div(pdf1.loc[:,'a1':'c2'].max(axis=1),axis=0)
# %%
idf1=idf1.melt()
idf1['cat']='iPSC'
pdf1=pdf1.melt()
pdf1['cat']='Partial'
df=pd.concat([idf1,pdf1],ignore_index=True)
# %%
fig=plt.figure(figsize=(20,10))
sns.boxplot(data=df,x='variable',y='value',hue='cat',showfliers = False)
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'-max-norm-coeff.svg'
plt.savefig(figname)
# %%
idf1=idf.copy()
pdf1=pdf.copy()
idf1=idf1.loc[:,'a1':'c2'].div(idf1.loc[:,'c1'],axis=0)
pdf1=pdf1.loc[:,'a1':'c2'].div(pdf1.loc[:,'c1'],axis=0)
# %%
idf1=idf1.melt()
idf1['cat']='iPSC'
pdf1=pdf1.melt()
pdf1['cat']='Partial'
df=pd.concat([idf1,pdf1],ignore_index=True)
# %%
fig=plt.figure(figsize=(20,10))
sns.boxplot(data=df,x='variable',y='value',hue='cat',showfliers = False)
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'-c1-norm-coeff.svg'
plt.savefig(figname)
# %%
if func=='XC_DI':
    A1=['a1','c1']
    A2=['a2','c2']
    R=['a1:a2','c1:c2']
else:
    A1=['a1','b1','c1']
    A2=['a2','b2','c2']
    R=['a1:a2','b1:b2','c1:c2']
# %%
irat=np.divide(idf.loc[:,A1],idf.loc[:,A2])
irat.columns=R
prat=np.divide(pdf.loc[:,A1],pdf.loc[:,A2])
prat.columns=R
# %%
idf1=irat.melt()
idf1['cat']='iPSC'
pdf1=prat.melt()
pdf1['cat']='Partial'
df=pd.concat([idf1,pdf1],ignore_index=True)
# %%
fig=plt.figure(figsize=(10,10))
sns.boxplot(data=df,x='variable',y='value',hue='cat',showfliers = False)
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'-ratio-coeff.svg'
plt.savefig(figname)
# %%
