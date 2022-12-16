#%%
import pandas as pd
import numpy as np
# %%
df=pd.read_excel('../input/iPSC_RawData.xlsx')
# %%
w_df=df.pivot_table(index=['ID','Timepoint'],columns=['Allele'],values='Allelic X to A ratio')
# %%
w_df=w_df.reset_index(level=0,drop=True)
#%%
w_df=w_df.apply(sorted,axis=1,raw=True,reverse=True)
w_df.columns=['Xa','Xi']
# %%
w_df=w_df.groupby(w_df.index).mean()
# %%
w_df.to_csv('../input/iPSC.csv')
# %%
