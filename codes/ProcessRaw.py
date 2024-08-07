#%%
import pandas as pd
import numpy as np
#%%
def process_data(Name):
    df = pd.read_excel('../input/'+Name+'_RawData.xlsx')
    df = df.pivot_table(index=['Cell ID','Timepoint'], columns=['allele'], values='Allelic X to A ratio')
    df = df.reset_index(level=0, drop=True)
    df = df.groupby(df.index).mean().reset_index()
    df['Timepoint'] = df['Timepoint'].str.replace('Day_', '')
    df.loc[df.Timepoint == 'iPSCs', 'Timepoint'] = 13
    df = df.astype({"Timepoint": int}).sort_values(by='Timepoint', ignore_index=True)
    df = df.rename(columns={'129S1': 'Xi', 'CAST': 'Xa'})
    return df
#%%
def partial_reactivation(w_df):
    wp_df = w_df.copy()
    last_t = wp_df.iloc[-1,0]
    wp_df.iloc[-1,:] = wp_df.iloc[-2,:]
    wp_df.iloc[-1,0] = last_t
    return wp_df
#%%
def interpol(df, tmin, tmax, ts=None):
    df = df.copy()
    if ts:
        df.iloc[0,0] = ts
    t=np.arange(tmin,tmax,1)
    Xi=np.interp(t,df['Timepoint'],df['Xi'])
    Xa=np.interp(t,df['Timepoint'],df['Xa'])
    df=pd.DataFrame({'t':t,'Xi':Xi,'Xa':Xa})
    return df
# %% Read the raw data
ipsc = process_data('iPSC')
interpol(ipsc, 0, 20).to_csv('../input/iPSC.csv',index=False)
# %% Timeshifted data
interpol(ipsc, 7, 16, ts=7).to_csv('../input/iPSC_timeshifted.csv',index=False)
# %% Partial reactivation data
partial = partial_reactivation(ipsc)
interpol(partial, 0, 20).to_csv('../input/Partial.csv',index=False)
# %% Partial timeshifted data
interpol(partial, 7, 16, ts=7).to_csv('../input/Partial_timeshifted.csv',index=False)
# %%
