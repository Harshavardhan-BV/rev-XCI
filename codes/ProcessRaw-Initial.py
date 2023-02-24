# %%
import pandas as pd
import numpy as np
# %% Read the raw data
df=pd.read_excel('../input/iPSC_RawData.xlsx')
df0=df[df.Timepoint=='Day_0']
# %% Pivot table by ID such that column is 129S1, CAST
w_df=df0.pivot_table(index=['ID','Timepoint'],columns=['Allele'],values='Allelic X to A ratio')
# %% Remove ID from index, only Timepoint is index
w_df=w_df.reset_index(drop=True)
# %%
w_df.columns=['Xi','Xa']
# %%
w_df.to_csv('../input/initial.csv',index=False)
# %%
