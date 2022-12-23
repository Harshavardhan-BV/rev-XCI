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
# idf['Cat']='iPSC'
# pdf['Cat']='Partial'
# df=pd.concat([idf,pdf],ignore_index=True)
# %%
fig=plt.figure(figsize=(20,10))
sns.boxplot(idf,showfliers = False,color='tab:red')
sns.boxplot(pdf,showfliers = False,color='tab:green')
plt.xlabel('Parameters')
plt.ylabel('Values')
figname='../figures/'+func+ts+'.svg'
plt.savefig(figname)
# %%
