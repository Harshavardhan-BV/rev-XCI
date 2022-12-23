# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=22
# %%
func='XC_DI'
# ts='_timeshifted'
ts=''
fname='../output/iPSC'+ts+'-'+func+'-parm.csv'
idf=pd.read_csv(fname)
fname='../output/Partial'+ts+'-'+func+'-parm.csv'
pdf=pd.read_csv(fname)
# %%
idf=idf.melt()
idf['cat']='iPSC'
pdf=pdf.melt()
pdf['cat']='Partial'
df=pd.concat([idf,pdf],ignore_index=True)
# %%
fig=plt.figure(figsize=(20,10))
sns.boxplot(data=df,x='variable',y='value',hue='cat',showfliers = False)
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'-parms.svg'
plt.savefig(figname)
