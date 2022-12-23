# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
plt.rcParams["svg.hashsalt"]=''
plt.rcParams["font.size"]=22
# %%
func='XC_DI'
ts='_timeshifted'
# ts=''
fname='../output/iPSC'+ts+'-'+func+'-parm.csv'
idf=pd.read_csv(fname)
fname='../output/Partial'+ts+'-'+func+'-parm.csv'
pdf=pd.read_csv(fname)
# %%
x=pd.concat([idf,pdf]).values
y=np.append(np.full(len(idf),'iPSC'),np.full(len(pdf),'Partial'))
# %%
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
PCs = pca.fit_transform(x)
# %%
df = pd.DataFrame(PCs, columns = ['PC1', 'PC2'])
df['cat']=y
# %%
fig=plt.figure(figsize=(10,10))
sns.scatterplot(df,x='PC1',y='PC2',hue='cat')
figname='../figures/'+func+ts+'-pca.svg'
plt.savefig(figname)