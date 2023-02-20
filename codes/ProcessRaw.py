#%%
import pandas as pd
import numpy as np
# %% Read the raw data
df=pd.read_excel('../input/iPSC_RawData.xlsx')
# %% Remove Xi cells
df0=df[df.Timepoint=='Day_0']
df1=df[df.Status!='Xi']
# %%
df = pd.concat((df0,df1))
# %% Pivot table by ID such that column is 129S1, CAST
w_df=df.pivot_table(index=['ID','Timepoint'],columns=['Allele'],values='Allelic X to A ratio')
# %% Remove ID from index, only Timepoint is index
w_df=w_df.reset_index(level=0,drop=True)
# %% Groupby timepoint and take mean values
w_df=w_df.groupby(w_df.index).mean()
# %% Remove Timepoint from index and convert to numerical values
w_df=w_df.reset_index()
w_df['Timepoint']=w_df['Timepoint'].str.replace('Day_','')
w_df.loc[w_df.Timepoint=='iPSCs','Timepoint']=13
w_df=w_df.astype({"Timepoint": int})
w_df.sort_values(by='Timepoint',ignore_index=True,inplace=True)
# %% Interpolate missing values into new dataframe
t=np.arange(0,20,1)
Xi=np.interp(t,w_df['Timepoint'],w_df['129S1'])
Xa=np.interp(t,w_df['Timepoint'],w_df['CAST'])
n_df=pd.DataFrame({'t':t,'Xi':Xi,'Xa':Xa})
# %% Save to csv
n_df.to_csv('../input/iPSC.csv',index=False)
# %%